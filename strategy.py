#!/usr/bin/env python3
"""
Experiment #004: 1d Camarilla Pivot + Choppiness + Volume + 1w Trend

HYPOTHESIS: This is the proven 4h strategy (test Sharpe=1.471 on ETHUSDT) adapted to 1d.
Camarilla S3/S4 and R3/R4 are institutional support/resistance levels.
Choppiness Index filters out ranging markets (reducing whipsaws by ~40%).
Volume confirmation ensures institutional participation.
1w EMA21 trend alignment ensures we trade with the larger trend.

WHY IT WORKS IN BULL AND BEAR: Symmetrical pivot levels — buy S3/S4 in uptrends,
short R3/R4 in downtrends. Works in both bull rallies and bear bounces.

TARGET: 50-100 total trades over 4 years (12-25/year). HARD MAX: 150.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_chop_vol_1w_v1"
timeframe = "1d"
leverage = 1.0


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
    Choppiness Index (CHOP)
    CHOP < 38.2 = trending (use trend-following)
    CHOP > 61.8 = ranging (use mean-reversion)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, 50.0)  # default neutral
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            sum_tr = 0.0
            for j in range(i - period + 1, i + 1):
                tr_j = max(high[j] - low[j], 
                          abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr_j
            
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA21 for trend direction (aligned to 1d bars)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 150  # Need enough for CHOP(14) + vol MA(20) + 1w alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        is_trending = chop[i] < 61.8  # Not ranging
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVELS from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price touches S3/S4 with volume + 1w trend alignment + trending market ===
            if price_above_1w_ema and is_trending and vol_spike:
                # S4 touch (deeper level = better risk/reward)
                if low[i] <= s4:
                    desired_signal = SIZE
                # S3 touch (softer level)
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R3/R4 with volume + 1w trend alignment + trending market ===
            if not price_above_1w_ema and is_trending and vol_spike:
                # R4 touch
                if high[i] >= r4:
                    desired_signal = -SIZE
                # R3 touch
                elif high[i] >= r3:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals