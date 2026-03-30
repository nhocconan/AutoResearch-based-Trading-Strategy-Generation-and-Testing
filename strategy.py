#!/usr/bin/env python3
"""
Experiment #028: 4h Williams%R + Donchian(16) + 1d SMA50 Trend + Choppiness

HYPOTHESIS: Williams %R at extremes (<-80 long, >-20 short) combined with 
Donchian(16) breakout captures institutional momentum. 1d SMA50 aligns with trend.
Choppiness < 55 keeps us out of range-bound markets. Simple entry = fewer trades.

WHY IT WORKS IN BULL AND BEAR:
- Bull: %R <-80 bounce + price above 1d SMA + Donchian breakout = momentum entry
- Bear: %R >-20 + price below 1d SMA + breakdown = short entry  
- Range: %R extremes catch reversals at channel boundaries

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 250.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_donchian_chop_1d_v3"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    wr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            wr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return wr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (16 periods = ~2.7 days on 4h)
    donchian_period = 16
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 100  # Need enough for Donchian(16) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME (Choppiness) ===
        # Only trade when trending (CHOP < 55), skip choppy markets
        is_trending = chop[i] < 55.0
        
        if is_trending:
            # === DONCHIAN BREAKOUT SIGNALS ===
            prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
            prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
            prev_close = close[i - 1] if i > 0 else close[i]
            
            # Williams %R extremes
            wr_oversold = williams_r[i] < -80  # Strong bullish momentum
            wr_overbought = williams_r[i] > -20  # Strong bearish momentum
            
            # Volume confirmation
            vol_spike = vol_ratio[i] > 1.5
            
            desired_signal = 0.0
            
            if not in_position:
                # === LONG: Breakout + oversold + uptrend ===
                if price_above_1d_sma and wr_oversold:
                    if high[i] > prev_donchian_high:
                        desired_signal = SIZE
                
                # === SHORT: Breakdown + overbought + downtrend ===
                if not price_above_1d_sma and wr_overbought:
                    if low[i] < prev_donchian_low:
                        desired_signal = -SIZE
            
            # === STOPLOSS CHECK (2.0 ATR trailing) ===
            stoploss_triggered = False
            
            if in_position and position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    stoploss_triggered = True
            
            if in_position and position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    stoploss_triggered = True
            
            if stoploss_triggered:
                desired_signal = 0.0
            
            # === CHOP EXIT: Exit when choppy ===
            if in_position and chop[i] >= 55.0:
                desired_signal = 0.0
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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