#!/usr/bin/env python3
"""
Experiment #021: 1d Donchian Breakout + 1w EMA + Volume Confirmation

HYPOTHESIS: 1d Donchian(21) breakout captures multi-day trend continuations.
Combined with 1w EMA(21) for trend alignment and volume confirmation,
this strategy catches institutional momentum while avoiding whipsaws.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Breakouts are symmetric: long breakouts above 1w EMA, short below
- 1w EMA filter ensures larger timeframe alignment
- Volume confirmation filters false breakouts
- ATR-based trailing stops adapt to volatility

TARGET: 75-120 total trades over 4 years (~6-8/year/symbol, 3 symbols).
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_ema_v1"
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
    
    # 1w EMA21 for trend alignment (larger TF bias)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Pre-compute 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (21-day = ~1 month)
    donchian_period = 21
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume MA (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND FILTER (1w EMA) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Price breaks above 21d high with volume + 1w trend
            if price_above_1w_ema and vol_spike:
                if close[i] > upper[i]:
                    desired_signal = SIZE
            
            # SHORT: Price breaks below 21d low with volume + 1w trend
            if not price_above_1w_ema and vol_spike:
                if close[i] < lower[i]:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (4 bars = 4 days to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # Exit on channel reversion (midpoint)
            mid = (upper[i] + lower[i]) / 2 if not np.isnan(upper[i]) and not np.isnan(lower[i]) else 0
            if position_side > 0 and close[i] > mid:
                desired_signal = 0.0
            if position_side < 0 and close[i] < mid:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals