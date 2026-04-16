#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary) ===
    # === 1w data (HTF trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # === 1w EMA50 (trend filter) ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1w Donchian channel (20-period) ===
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous week's channel)
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # === 1w volume ratio for confirmation ===
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = volume_1w / vol_ma_20_1w
    
    # === 14-period ATR for stoploss ===
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ratio_1w[i]) or
            np.isnan(atr_14_1w[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_ratio = vol_ratio_1w[i]
        atr = atr_14_1w[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.5 * ATR
            if price < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.5 * ATR
            if price > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend reverses (below EMA50)
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend reverses (above EMA50)
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian upper with volume, in uptrend (above EMA50)
            if price > upper and vol_ratio > 2.0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below Donchian lower with volume, in downtrend (below EMA50)
            elif price < lower and vol_ratio > 2.0 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_1wEMA50_Volume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0