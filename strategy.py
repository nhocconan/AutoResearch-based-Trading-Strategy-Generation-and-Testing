#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Mean Reversion + Volume + EMA Trend

HYPOTHESIS: After Donchian breakout, price often mean-reverts to the middle band.
By fading extremes (buying at lower band, shorting at upper band) with trend 
confirmation AND volume, we catch reversals at key levels.

WHY IT WORKS IN BULL AND BEAR: Mean reversion to the middle band works in both
directions — buy support at lower band in uptrends, short resistance at upper
band in downtrends. 4h timeframe gives ~75-200 trades over 4 years (proven range).

LEARNED FROM FAILURES: Keep conditions simple (2-3 max), don't stack filters
that never align. Volume confirmation is enough without choppiness (which often
contradicts entry signals).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_meanrev_vol_1d_v1"
timeframe = "4h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - uses past period values, no look-ahead"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend (align to 4h bars)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    upper, lower, middle = calculate_donchian(high, low, period=20)
    
    # Volume ratio
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
    
    warmup = 50  # ATR and Donchian warmup
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        trend_up = close[i] > ema_1d_aligned[i]
        trend_down = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.2
        
        # Donchian bands
        upper_band = upper[i]
        lower_band = lower[i]
        middle_band = middle[i]
        
        if np.isnan(upper_band):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price at/exceeds lower band + trend up + volume ===
            if trend_up and vol_spike:
                if low[i] <= lower_band:
                    desired_signal = SIZE
            
            # === SHORT: Price at/exceeds upper band + trend down + volume ===
            if trend_down and vol_spike:
                if high[i] >= upper_band:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === EXIT: Price reaches middle band (mean reversion complete) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            if position_side > 0 and close[i] >= middle_band:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= middle_band:
                desired_signal = 0.0
        
        # === PARTIAL PROFIT (half at 2R profit) ===
        if in_position and position_side > 0:
            profit_pct = (close[i] - entry_price) / entry_price
            if profit_pct >= 0.05:  # 5% = ~2R if ATR ~2.5%
                desired_signal = SIZE * 0.5
        
        if in_position and position_side < 0:
            profit_pct = (entry_price - close[i]) / entry_price
            if profit_pct >= 0.05:
                desired_signal = -SIZE * 0.5
        
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals