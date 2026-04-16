#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Breakout with Volume & Trend Confirmation ===
    # Long: Close > Donchian(20) High + Volume Spike + 4h Close > 4h EMA34
    # Short: Close < Donchian(20) Low + Volume Spike + 4h Close < 4h EMA34
    
    # Get 4h data for Donchian and EMA
    df_4h = get_htf_data(prices, '4h')
    
    # Donchian(20) on 4h
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # EMA34 on 4h
    ema34 = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe (primary)
    donch_high_4h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_4h, donch_low)
    ema34_4h = align_htf_to_ltf(prices, df_4h, ema34)
    
    # Volume confirmation (on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup for 4h calculations
    warmup = 80  # 20 for Donchian + 34 for EMA + buffer
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(ema34_4h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit on Donchian low break or trend reversal
            if price < donch_low_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit on Donchian high break or trend reversal
            if price > donch_high_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian high with volume + trend
            if price > donch_high_4h[i] and vol_spike and close[i] > ema34_4h[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Break below Donchian low with volume + trend
            elif price < donch_low_4h[i] and vol_spike and close[i] < ema34_4h[i]:
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

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0