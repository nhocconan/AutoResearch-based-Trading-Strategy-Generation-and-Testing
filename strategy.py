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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR on 6h (for volatility filter)
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate 1d ATR (for Donchian band width)
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w EMA(20) for long-term trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_6h / np.where(vol_ma_20 > 0, vol_ma_20, 1)
    vol_ratio_aligned = vol_ratio  # already on 6h timeframe
    
    # Calculate 6-day Donchian channels on 1d data (20 periods)
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        ema_50_val = ema_50_1d_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        atr_6h_val = atr_6h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 1d EMA(50) OR Donchian low breaks
            if (price < ema_50_val) or (price < donchian_low):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 1d EMA(50) OR Donchian high breaks
            if (price > ema_50_val) or (price > donchian_high):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Calculate Donchian width normalized by ATR
            donchian_width = donchian_high - donchian_low
            norm_width = donchian_width / atr_1d_val if atr_1d_val > 0 else 0
            
            # LONG: Price breaks above Donchian high with volume surge AND
            # 1d EMA(50) above 1w EMA(20) (bullish alignment) AND
            # Donchian width not too tight (avoid chop)
            if (price > donchian_high) and (vol_ratio_val > 1.5) and \
               (ema_50_val > ema_20_1w_val) and (norm_width > 1.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian low with volume surge AND
            # 1d EMA(50) below 1w EMA(20) (bearish alignment) AND
            # Donchian width not too tight (avoid chop)
            elif (price < donchian_low) and (vol_ratio_val > 1.5) and \
                 (ema_50_val < ema_20_1w_val) and (norm_width > 1.0):
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

name = "6h_DonchianBreakout_VolumeSurge_TrendAlign"
timeframe = "6h"
leverage = 1.0