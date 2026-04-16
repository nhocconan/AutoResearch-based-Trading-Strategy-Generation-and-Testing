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
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h ATR (14) for position sizing and stops ===
    tr_4h = np.maximum(high_4h - low_4h,
                       np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                  np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 1d ATR (14) for volatility regime ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h Moving Average (50) for trend filter ===
    ma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    ma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ma_50_4h)
    
    # === 1d Bollinger Bands (20, 2) for volatility regime ===
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # === 4h Volume spike detection (volume > 1.5x 20-period MA) ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ma_50_4h_aligned[i]) or np.isnan(bb_width_1d_aligned[i]) or 
            np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_4h[i]
        atr_4h_val = atr_4h_aligned[i]
        ma_50_val = ma_50_4h_aligned[i]
        bb_width_val = bb_width_1d_aligned[i]
        vol_ratio_val = vol_ratio_4h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below MA(50) OR volatility regime shifts to high
            if (price < ma_50_val) or (bb_width_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above MA(50) OR volatility regime shifts to high
            if (price > ma_50_val) or (bb_width_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above MA(50) AND low volatility regime AND volume spike
            if (price > ma_50_val) and (bb_width_val < 30) and (vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below MA(50) AND low volatility regime AND volume spike
            elif (price < ma_50_val) and (bb_width_val < 30) and (vol_ratio_val > 1.5):
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

name = "4h_MA50_LowVol_Volume_Spike"
timeframe = "4h"
leverage = 1.0