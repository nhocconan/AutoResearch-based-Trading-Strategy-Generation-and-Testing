#!/usr/bin/env python3
"""
Experiment #023: 1d Donchian(20) Breakout + 1w EMA Trend + Volume

HYPOTHESIS: Daily Donchian(20) captures medium-term breakout momentum.
Combined with weekly trend alignment (above = bullish, below = bearish),
volume confirmation, and ATR stoploss.

WHY IT WORKS IN BULL AND BEAR: Symmetrical breakout structure — buy breakouts
above weekly EMA in uptrends, short breakouts below weekly EMA in downtrends.
1d timeframe naturally limits trade frequency.

TARGET: 50-100 total over 4 years (12-25/year). HARD MAX: 150.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily indicators (computed once, outside loop)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20-day)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume ratio (20-day MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 200  # Donchian(20) + EMA buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === Weekly trend direction ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price breaks above 20-day high + bullish weekly trend + volume ===
            if price_above_1w_ema and vol_spike:
                if high[i] > donchian_high_20[i]:
                    desired_signal = SIZE
            
            # === SHORT: Price breaks below 20-day low + bearish weekly trend + volume ===
            if not price_above_1w_ema and vol_spike:
                if low[i] < donchian_low_20[i]:
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
        
        # === HOLD PERIOD: minimum 3 bars to avoid churn ===
        bars_held = i - entry_bar
        if bars_held < 3:
            if in_position and desired_signal == 0.0:
                desired_signal = position_side * SIZE
        
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
        
        signals[i] = desired_signal
    
    return signals