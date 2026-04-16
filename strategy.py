#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === Volume Confirmation (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 4h EMA Trend Filter (34-period) ===
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100  # Need EMA34 and ATR14 data
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema34 = ema_34_4h_aligned[i]
        atr = atr_14_aligned[i]
        
        # === STOP LOSS: Exit when price moves against position by 2.5 * ATR ===
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC: Exit when price returns to midline (average of Donchian) ===
        midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        
        if position == 1:  # Long position
            # Exit when price crosses back below midline (failed bullish continuation)
            if price < midline:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above midline (failed bearish continuation)
            if price > midline:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and price > EMA34
            if price > donchian_high_aligned[i] and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price breaks below Donchian low with volume confirmation and price < EMA34
            elif price < donchian_low_aligned[i] and vol_spike and price < ema34:
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

name = "4h_Donchian20_4h_Volume_EMA34_ATR25Stop"
timeframe = "4h"
leverage = 1.0