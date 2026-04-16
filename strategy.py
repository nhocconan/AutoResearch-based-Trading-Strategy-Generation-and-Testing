# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Works in bull markets via breakout momentum and in bear via trend-filtered short entries
# Target: 20-40 trades/year to avoid fee drag; uses discrete sizing (0.25) for low turnover

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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === 4h Donchian channel (20-period) ===
    donchian_high = np.zeros_like(close_4h)
    donchian_low = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i < 20:
            donchian_high[i] = np.max(high_4h[:i+1])
            donchian_low[i] = np.min(low_4h[:i+1])
        else:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # === 12h EMA (34-period) for trend filter ===
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h volume ratio (20-period) ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    # Align all HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and EMA
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        ema = ema_12h_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish
            if price < dl or close[i] < ema:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish
            if price > dh or close[i] > ema:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Donchian breakout above high with volume and bullish trend
            if price > dh and vol_ratio > 1.5 and close[i] > ema:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Donchian breakdown below low with volume and bearish trend
            elif price < dl and vol_ratio > 1.5 and close[i] < ema:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_EMA34_Volume_Filter"
timeframe = "4h"
leverage = 1.0