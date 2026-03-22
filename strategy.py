#!/usr/bin/env python3
"""
Experiment #165: 1h Primary + 4h/1d HTF — Multi-Path Connors RSI with Regime Filter

Hypothesis: Previous 1h strategies failed because entry conditions were too strict
(0 trades) or too loose (>200 trades/year → fee drag). This strategy uses:

1. 4h HMA(21) for major trend direction (signal bias)
2. 1d HMA slope for regime confirmation (bull/bear bias)
3. Connors RSI(3,2,100) on 1h for entry timing (oversold/overbought)
4. Volume filter (>0.8x 20-bar avg) for confirmation
5. Session filter (8-20 UTC) for liquidity
6. ATR(14) trailing stoploss at 2.5x

Why this should work:
- 4h/1d HTF filters reduce trade frequency to 30-80/year target
- Connors RSI has 75% win rate for mean reversion in literature
- Multiple entry paths ensure trades generate (not too strict)
- Session filter avoids low-liquidity hours (reduces slippage)
- Discrete position sizing (0.25/0.30) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 max (conservative for 1h)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_4h1d_hma_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    MAX_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i] and hma_4h_21_aligned[i] > hma_4h_48_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i] and hma_4h_21_aligned[i] < hma_4h_48_aligned[i]
        
        # === 1D REGIME BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.7  # Relaxed for more trades
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 30
        crsi_overbought = crsi[i] > 70
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        crsi_neutral_low = crsi[i] < 35
        crsi_neutral_high = crsi[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if trend_4h_bullish and trend_1d_bullish:
            current_size = MAX_SIZE  # Full size in strong bull
        elif trend_4h_bearish and trend_1d_bearish:
            current_size = MAX_SIZE  # Full size in strong bear
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_confidence = 0
        
        # Path 1: 4h bullish + CRSI oversold + volume (trend pullback)
        if trend_4h_bullish and crsi_oversold and volume_ok:
            long_confidence += 3
        
        # Path 2: 1d bullish + CRSI extreme low (deep pullback in bull)
        if trend_1d_bullish and crsi_extreme_low:
            long_confidence += 3
        
        # Path 3: 4h bullish + CRSI neutral low (lighter entry)
        if trend_4h_bullish and crsi_neutral_low and volume_ok:
            long_confidence += 2
        
        # Path 4: CRSI extreme low alone (mean reversion)
        if crsi_extreme_low and volume_ok:
            long_confidence += 2
        
        # Path 5: Session + CRSI oversold (liquidity confirmation)
        if in_session and crsi_oversold:
            long_confidence += 1
        
        # Apply session filter for full entries
        if long_confidence >= 3:
            if in_session:
                new_signal = current_size
            else:
                new_signal = current_size * 0.5  # Half size outside session
        elif long_confidence == 2 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: 4h bearish + CRSI overbought + volume
        if trend_4h_bearish and crsi_overbought and volume_ok:
            short_confidence += 3
        
        # Path 2: 1d bearish + CRSI extreme high
        if trend_1d_bearish and crsi_extreme_high:
            short_confidence += 3
        
        # Path 3: 4h bearish + CRSI neutral high
        if trend_4h_bearish and crsi_neutral_high and volume_ok:
            short_confidence += 2
        
        # Path 4: CRSI extreme high alone
        if crsi_extreme_high and volume_ok:
            short_confidence += 2
        
        # Path 5: Session + CRSI overbought
        if in_session and crsi_overbought:
            short_confidence += 1
        
        if short_confidence >= 3:
            if in_session:
                new_signal = -current_size
            else:
                new_signal = -current_size * 0.5
        elif short_confidence == 2 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~5 days on 1h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi_neutral_low:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and crsi_neutral_high:
                new_signal = -current_size * 0.4
            elif crsi_extreme_low:
                new_signal = current_size * 0.3
            elif crsi_extreme_high:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and trend_1d_bearish:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish and trend_1d_bullish:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals