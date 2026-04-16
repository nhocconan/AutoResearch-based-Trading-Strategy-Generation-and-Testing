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
    
    # === Daily data for 1d ATR and volume ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Calculate 14-day ATR on daily data ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First day
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Calculate 50-period EMA on weekly close ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Align ATR and EMA to daily timeframe ===
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily volume ratio (volume / 20-day average) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup - need enough data for ATR and EMA
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_val = atr_14_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below EMA50 weekly (trend change) or ATR-based stop
            if price < ema_50_1w_val or price < close[i-1] - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA50 weekly (trend change) or ATR-based stop
            if price > ema_50_1w_val or price > close[i-1] + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above weekly EMA50 (uptrend) + volume spike + momentum
            if (price > ema_50_1w_val) and (vol_ratio_val > 2.0) and (close[i] > close[i-1]):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below weekly EMA50 (downtrend) + volume spike + momentum
            elif (price < ema_50_1w_val) and (vol_ratio_val > 2.0) and (close[i] < close[i-1]):
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

name = "1d_EMA50_Trend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0