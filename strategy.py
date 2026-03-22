#!/usr/bin/env python3
"""
Experiment #170: 1h Primary + 4h/12h HTF — Vol Spike Mean Reversion + Multi-Confluence

Hypothesis: Previous 1h strategies failed due to (1) too many trades causing fee drag,
or (2) too strict filters causing 0 trades. This strategy uses:

1. 4h/12h HMA trend bias for signal DIRECTION (not fighting major trend)
2. Volatility spike detection (ATR7/ATR30 > 1.5) for entry TIMING
3. Connors RSI extremes (<20 long, >80 short) for mean reversion entries
4. Bollinger Band confirmation (price beyond 2.25 std)
5. Session filter (8-20 UTC) to avoid low-liquidity periods
6. Volume confirmation (>0.8x 20-bar avg) to confirm real moves

Why this should work:
- 1h timeframe with HTF direction = fewer trades than pure 1h, better timing than 4h
- Vol spike + BB extreme = capitulation events (high win rate reversals)
- Session filter reduces noise from Asian overnight hours
- Discrete position sizing (0.25) minimizes fee churn
- Target: 40-70 trades/year (within 30-80 target for 1h)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF to reduce fee impact)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_volspike_connors_bb_4h12h_session_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.25):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
            streak_rsi[i] = min(100, 50 + streak[i] * 12)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 12)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.25)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume moving average
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    # UTC hour for session filter
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_slope_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(vol_sma20[i]) or vol_sma20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === HTF TREND BIAS (4h + 12h confluence) ===
        # 12h major trend
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # 4h intermediate trend
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.3
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.3
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA alignment (21 > 48 = bullish structure)
        hma_4h_bullish_struct = hma_4h_21_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish_struct = hma_4h_21_aligned[i] < hma_4h_48_aligned[i]
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.5
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 18
        crsi_extreme_high = crsi[i] > 82
        
        # === VOLUME CONFIRMATION ===
        vol_confirmation = volume[i] > 0.8 * vol_sma20[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size outside session hours
        if not in_session:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_confidence = 0
        
        # Path 1: Vol spike + BB lower + CRSI oversold (capitulation) - HIGH CONFIDENCE
        if vol_spike and price_below_bb_lower and crsi_oversold:
            long_score += 4
            long_confidence += 2
        
        # Path 2: 12h bullish + 4h bullish + CRSI pullback
        if trend_12h_bullish and trend_4h_bullish and crsi[i] < 35:
            long_score += 3
            long_confidence += 1
        
        # Path 3: Price below 12h HMA but CRSI very low (deep pullback in bull)
        if price_above_12h_hma and crsi_extreme_low:
            long_score += 3
            long_confidence += 1
        
        # Path 4: 4h bullish structure + BB lower + volume
        if hma_4h_bullish_struct and price_below_bb_lower and vol_confirmation:
            long_score += 3
            long_confidence += 1
        
        # Path 5: CRSI extreme + vol spike (fallback for more trades)
        if crsi_extreme_low and vol_spike:
            long_score += 2
        
        # Path 6: Simple oversold in session (more trades)
        if in_session and crsi[i] < 22 and price_below_bb_lower:
            long_score += 2
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score >= 3 and bars_since_last_trade > 48:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 72:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Vol spike + BB upper + CRSI overbought
        if vol_spike and price_above_bb_upper and crsi_overbought:
            short_score += 4
            short_confidence += 2
        
        # Path 2: 12h bearish + 4h bearish + CRSI pullback
        if trend_12h_bearish and trend_4h_bearish and crsi[i] > 65:
            short_score += 3
            short_confidence += 1
        
        # Path 3: Price below 12h HMA but CRSI very high (rally in bear)
        if price_below_12h_hma and crsi_extreme_high:
            short_score += 3
            short_confidence += 1
        
        # Path 4: 4h bearish structure + BB upper + volume
        if hma_4h_bearish_struct and price_above_bb_upper and vol_confirmation:
            short_score += 3
            short_confidence += 1
        
        # Path 5: CRSI extreme + vol spike
        if crsi_extreme_high and vol_spike:
            short_score += 2
        
        # Path 6: Simple overbought in session
        if in_session and crsi[i] > 78 and price_above_bb_upper:
            short_score += 2
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score >= 3 and bars_since_last_trade > 48:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 72:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~8 days on 1h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and crsi[i] < 35:
                new_signal = current_size * 0.4
            elif trend_12h_bearish and crsi[i] > 65:
                new_signal = -current_size * 0.4
            elif crsi[i] < 20 and in_session:
                new_signal = current_size * 0.3
            elif crsi[i] > 80 and in_session:
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_12h_bearish and trend_4h_bearish:
                trend_reversal = True
            if position_side < 0 and trend_12h_bullish and trend_4h_bullish:
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