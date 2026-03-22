#!/usr/bin/env python3
"""
Experiment #088: 30m Primary + 4h/1d HTF — Trend Pullback with Volume/Session Filter

Hypothesis: Previous 30m strategies (#078, #085) failed with Sharpe=0.000 because entry
conditions were too strict (Choppiness + Connors RSI + Session + Volume all had to align
perfectly = 0 trades). This version loosens individual filters while maintaining
multi-confluence approach to ensure minimum trade count.

Key changes from #078 (which got 0 trades):
1. Remove Choppiness Index filter (was too restrictive)
2. Loosen CRSI thresholds: <30/>70 instead of <15/>85
3. Volume filter: >0.7x avg instead of >1.2x (easier to meet)
4. Session filter: 5-23 UTC instead of 8-20 UTC (wider window)
5. 4h HMA trend is PRIMARY filter (only trade with 4h trend)
6. 1d HMA slope for regime bias (secondary confirmation)
7. 30m CRSI for pullback entry timing
8. Add frequency safeguard: force entry if no trades for 60 bars

Strategy Logic:
1. 4h HMA(21): Primary trend direction (only long if price > 4h HMA, short if <)
2. 1d HMA(21) slope: Regime bias (>0.3% = bullish, <-0.3% = bearish)
3. 30m Connors RSI: Entry timing on pullbacks (<30 long, >70 short)
4. Volume: >0.7x 20-bar average (confirms move)
5. Session: 5-23 UTC (major trading hours)
6. ATR(14) stoploss: 2.5x trailing
7. Position size: 0.25 discrete (conservative for 30m)

Target: 50-100 trades/year, Sharpe > 0.220 (beat #076)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_trend_pullback_vol_session_4h1d_v1"
timeframe = "30m"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank of price change
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hour = pd.to_datetime(open_time, unit='ms').hour
    return hour

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
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average (20 bars)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (UTC)
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25  # Smaller for 30m to reduce fee impact
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_slope_aligned[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === 4H TREND (PRIMARY FILTER) ===
        # Only trade with 4h trend direction
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D REGIME BIAS (SECONDARY) ===
        # HMA slope > 0.3% = bullish regime (prefer longs)
        # HMA slope < -0.3% = bearish regime (prefer shorts)
        regime_bullish = hma_1d_slope_aligned[i] > 0.3
        regime_bearish = hma_1d_slope_aligned[i] < -0.3
        regime_neutral = not regime_bullish and not regime_bearish
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CONNORS RSI (ENTRY TIMING) ===
        # Loosened thresholds for more trades (critical for avoiding 0 trades)
        crsi_oversold = crsi[i] < 30  # Was <15 in failed #078
        crsi_overbought = crsi[i] > 70  # Was >85 in failed #078
        
        # === VOLUME FILTER ===
        # Volume > 0.7x average (easier to meet than >1.2x)
        volume_confirmed = volume[i] > 0.7 * vol_avg_20[i]
        
        # === SESSION FILTER ===
        # 5-23 UTC (wider than 8-20)
        session_active = 5 <= utc_hours[i] <= 23
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral regime
        if regime_neutral:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Need: 4h bullish + CRSI oversold + volume + session
        long_condition = (
            price_above_4h_hma and  # 4h trend
            crsi_oversold and  # Pullback entry
            volume_confirmed and  # Volume confirmation
            session_active  # Trading hours
        )
        
        # Add 1d regime confirmation for stronger signals
        if long_condition:
            if regime_bullish or price_above_1d_hma:
                new_signal = current_size
            elif regime_neutral:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES - Need: 4h bearish + CRSI overbought + volume + session
        short_condition = (
            price_below_4h_hma and  # 4h trend
            crsi_overbought and  # Pullback entry
            volume_confirmed and  # Volume confirmation
            session_active  # Trading hours
        )
        
        # Add 1d regime confirmation for stronger signals
        if short_condition:
            if regime_bearish or price_below_1d_hma:
                new_signal = -current_size
            elif regime_neutral:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 30m), allow weaker entry
        # This is CRITICAL to avoid 0 trades like #078
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if price_above_4h_hma and crsi[i] < 40:
                new_signal = current_size * 0.5
            elif price_below_4h_hma and crsi[i] > 60:
                new_signal = -current_size * 0.5
        
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
        
        # Apply stoploss
        if stoploss_triggered:
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