#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume spike filter (volume > 1.8x 20-period median) and 1d EMA trend filter
# Long when price > Camarilla R1 AND 4h volume > 1.8x 20h median volume AND 1d close > 1d EMA50
# Short when price < Camarilla S1 AND 4h volume > 1.8x 20h median volume AND 1d close < 1d EMA50
# Exit when price crosses Camarilla pivot point (PP)
# Uses discrete position size 0.20. Target: 60-150 total trades over 4 years (15-37/year).
# Combines intraday price structure with volume confirmation and daily trend filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Camarilla levels and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla pivot points (R1, S1, PP) and volume median ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Camarilla pivot points calculation
    camarilla_pp = (high_4h + low_4h + close_4h) / 3.0
    camarilla_r1 = camarilla_pp + (high_4h - low_4h) * 1.1 / 12.0
    camarilla_s1 = camarilla_pp - (high_4h - low_4h) * 1.1 / 12.0
    
    # Volume median for scaling
    volume_median_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1h)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    volume_median_aligned = align_htf_to_ltf(prices, df_4h, volume_median_20_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align 4h volume for volume confirmation
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 4h Camarilla, 4h volume median, 1d EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_median_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_median = volume_median_aligned[i]
        daily_ema50 = ema_50_1d_aligned[i]
        vol_4h = volume_4h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot point (mean reversion to PP)
            if price < pp:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot point (mean reversion to PP)
            if price > pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: current 4h volume > 1.8x 20h median volume
            vol_threshold = vol_median * 1.8
            vol_confirm = vol_4h > vol_threshold
            
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume confirmation AND daily uptrend
            if price > r1 and vol_confirm and price > daily_ema50:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume confirmation AND daily downtrend
            elif price < s1 and vol_confirm and price < daily_ema50:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_Camarilla_R1S1_4hVolumeSpike1.8x_1dEMA50_v1"
timeframe = "1h"
leverage = 1.0