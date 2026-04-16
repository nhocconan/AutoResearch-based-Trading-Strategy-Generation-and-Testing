#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 1d volume confirmation and weekly trend filter
# Long when price > Camarilla R1 AND 1d volume > 1.5x 20-period median volume AND weekly close > weekly EMA50
# Short when price < Camarilla S1 AND 1d volume > 1.5x 20-period median volume AND weekly close < weekly EMA50
# Exit when price crosses Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete position size 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
# Combines intraday price levels with volume confirmation and weekly trend filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla levels (R1, S1, PP) and Volume median (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla levels calculation
    rango = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = camarilla_pp + (rango * 1.1 / 12)
    camarilla_s1 = camarilla_pp - (rango * 1.1 / 12)
    
    # Volume median
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 1d Camarilla, 1d volume, weekly EMA
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(vol_median_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pp = camarilla_pp_aligned[i]
        vol_median = vol_median_20_1d_aligned[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume median
        vol_threshold = vol_median * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Weekly trend filter
        weekly_trend_up = close_1d[-1] > weekly_ema if len(close_1d) > 0 else False  # Use latest known weekly close
        # Actually, we need the weekly close value aligned to current 6h bar
        df_1w_close = get_htf_data(prices, '1w')['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w_close)
        if np.isnan(weekly_close_aligned[i]):
            signals[i] = 0.0
            continue
        weekly_trend_up = weekly_close_aligned[i] > weekly_ema
        weekly_trend_down = weekly_close_aligned[i] < weekly_ema
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot point (mean reversion)
            if price < pp:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot point (mean reversion)
            if price > pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume confirmation AND weekly uptrend
            if price > r1 and vol_confirm and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume confirmation AND weekly downtrend
            elif price < s1 and vol_confirm and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_Camarilla_R1S1_1dVolumeConfirm_1wTrend_v1"
timeframe = "6h"
leverage = 1.0