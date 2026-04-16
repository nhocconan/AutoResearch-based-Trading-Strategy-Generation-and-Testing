#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF for trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1d Williams Alligator (Jaws, Teeth, Lips) ===
    median_price_1d = (high_1d + low_1d) / 2
    # Jaws: SMA(13) of median price, shifted 8 bars forward
    jaws = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8) of median price, shifted 5 bars forward
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5) of median price, shifted 3 bars forward
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components with proper delay for forward shifts
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1w EMA(34) for trend ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d volume ratio for confirmation ===
    vol_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = volume_1d / vol_ma_10_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Alligator and EMA
    warmup = 40
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        ema_1w = ema_34_1w_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: Alligator lines cross bearish or price below weekly EMA
            if jaw < tooth < lip or price < ema_1w:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross bullish or price above weekly EMA
            if jaw > tooth > lip or price > ema_1w:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Alligator alignment: bullish (jaw > tooth > lip) or bearish (jaw < tooth < lip)
            # Price relative to weekly EMA for trend filter
            # Volume confirmation: above average
            if jaw > tooth > lip and price > ema_1w and vol_ratio > 1.3:
                # LONG: Alligator bullish, price above weekly EMA, volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif jaw < tooth < lip and price < ema_1w and vol_ratio > 1.3:
                # SHORT: Alligator bearish, price below weekly EMA, volume confirmation
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

name = "1d_Alligator_EMA34_1w_Volume"
timeframe = "1d"
leverage = 1.0