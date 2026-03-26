#!/usr/bin/env python3
"""
Experiment #004: 1d Primary + 1w HTF — Camarilla Pivot + Volume Spike

Hypothesis: Daily timeframe with weekly trend bias is optimal because:
1. 1d bars capture the "big moves" that matter - avoids chop on lower TFs
2. Weekly KAMA filters out counter-trend trades (bear market shorts only)
3. Camarilla S3/R3 levels are proven support/resistance on daily charts
4. Volume spike confirms institutional interest at pivot levels
5. Target 40-80 trades over 4 years = 10-20/year = minimal fee drag

Why this should work in BOTH bull and bear:
- Bull: Price respects R3 breakout with volume = continuation
- Bear: Weekly KAMA confirms downtrend, short R3 touches with volume
- Range: Choppiness filter prevents whipsaw trades

Key design (based on DB winner gen_camarilla_pivot_volume_spike_choppiness_4h_v1):
- Simplified to 1d (original was 4h)
- Weekly KAMA instead of 4h HMA (more stable trend filter)
- Camarilla S3/R3 breakout (most reliable pivot levels)
- Volume spike confirmation (>1.5x 20d avg = institutional interest)
- Choppiness regime to avoid ranging markets

Target: Sharpe>0.5, trades 40-80 train, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_volume_spike_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=30, fast_ema=2, slow_ema=30):
    """
    Kaufman Adaptive Moving Average - adapts to volatility
    period: lookback for efficiency ratio
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate price changes
    change = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate volatility (sum of price changes)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(change[i - period + 1:i + 1])
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 1e-10:
            er[i] = change[i] / volatility[i]
    
    # Smoothing constants
    fast = 2 / (fast_ema + 1)
    slow = 2 / (slow_ema + 1)
    smoothing = er * (fast - slow) + slow
    
    # Square the smoothing
    smoothing = smoothing ** 2
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]) and not np.isnan(smoothing[i]):
            kama[i] = kama[i - 1] + smoothing[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_camarilla_pivots(high, low, close, period=1):
    """
    Camarilla Pivot Levels - 8 levels based on yesterday's range
    Focus on S3/R3 as primary entry points (most reliable)
    
    R3 = close + (high - low) * 1.1
    R2 = close + (high - low) * 1.1/2
    R1 = close + (high - low) * 1.1/4
    S1 = close - (high - low) * 1.1/4
    S2 = close - (high - low) * 1.1/2
    S3 = close - (high - low) * 1.1
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    r3 = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        prev_close = close[i - period]
        prev_high = high[i - period]
        prev_low = low[i - period]
        prev_range = prev_high - prev_low
        
        if prev_range > 1e-10:
            r3[i] = prev_close + prev_range * 1.1
            r1[i] = prev_close + prev_range * 1.1 / 4
            s1[i] = prev_close - prev_range * 1.1 / 4
            s3[i] = prev_close - prev_range * 1.1
    
    return r3, r1, s1, s3

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """
    Volume spike detection: current volume > threshold * SMA(volume)
    Returns: 1 = spike up, 0 = normal, -1 = spike down
    """
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            ratio = volume[i] / vol_sma[i]
            if ratio > threshold:
                spike[i] = 1.0  # volume spike up
            elif ratio < (1.0 / threshold):
                spike[i] = -1.0  # volume dry up
    
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align weekly indicators
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate daily indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    r3, r1, s1, s3 = calculate_camarilla_pivots(high, low, close, period=1)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(r3[i]) or np.isnan(s3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND DIRECTION (1w KAMA bias) ===
        weekly_bullish = close[i] > kama_1w_aligned[i]
        weekly_bearish = close[i] < kama_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Neutral to trending threshold
        
        # === CAMARILLA LEVELS ===
        r3_prev = r3[i-1] if i > 0 and not np.isnan(r3[i-1]) else r3[i]
        r1_prev = r1[i-1] if i > 0 and not np.isnan(r1[i-1]) else r1[i]
        s1_prev = s1[i-1] if i > 0 and not np.isnan(s1[i-1]) else s1[i]
        s3_prev = s3[i-1] if i > 0 and not np.isnan(s3[i-1]) else s3[i]
        
        # === VOLUME SPIKE ===
        has_vol_spike = vol_spike[i] > 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + price breaks R3 + volume spike
        if weekly_bullish and is_trending:
            # Price approaching or breaking R3 with volume
            price_near_r3 = close[i] >= r3_prev * 0.995
            price_at_r1 = close[i] >= r1_prev * 0.99
            
            if price_near_r3 and has_vol_spike:
                desired_signal = SIZE_STRONG
            elif price_at_r1 and has_vol_spike:
                desired_signal = SIZE_BASE
        
        # SHORT: Weekly bearish + price breaks S3 + volume spike
        elif weekly_bearish and is_trending:
            # Price approaching or breaking S3 with volume
            price_near_s3 = close[i] <= s3_prev * 1.005
            price_at_s1 = close[i] <= s1_prev * 1.01
            
            if price_near_s3 and has_vol_spike:
                desired_signal = -SIZE_STRONG
            elif price_at_s1 and has_vol_spike:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals