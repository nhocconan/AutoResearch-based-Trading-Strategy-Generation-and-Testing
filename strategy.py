#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike confirmation and 1w EMA50 trend filter.
# Long when price > R1, 1d volume > 2.0x its 20-period median, and weekly close > weekly EMA50.
# Short when price < S1, same volume spike condition, and weekly close < weekly EMA50.
# Exit when price crosses the Camarilla pivot point (mean reversion).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Combines price channel breakout with volume spike filter and weekly trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla pivot levels (based on previous day) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC (requires daily data for proper calculation)
    # We'll use 1d data to get proper OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # Pivot = (High + Low + Close) / 3
    rang = high_1d - low_1d
    camarilla_r1 = close_1d + rang * 1.1 / 12
    camarilla_s1 = close_1d - rang * 1.1 / 12
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align daily volume for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # daily Camarilla, daily volume median, weekly EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        vol_median = vol_median_aligned[i]
        weekly_ema50 = ema_50_1w_aligned[i]
        daily_volume = vol_1d_aligned[i]
        
        # Get aligned daily close for proper volume comparison and trend
        df_1d_close = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d_close)
        weekly_trend_up = daily_close_aligned[i] > weekly_ema50  # Using daily close vs weekly EMA for trend
        weekly_trend_down = daily_close_aligned[i] < weekly_ema50
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot (mean reversion)
            if price < pivot:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot (mean reversion)
            if price > pivot:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current daily volume > 2.0x its 20-period median
            volume_spike = daily_volume > (vol_median * 2.0)
            
            # LONG CONDITIONS
            # Price breaks above Camarilla R1 AND volume spike AND weekly uptrend
            if price > r1 and volume_spike and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S1 AND volume spike AND weekly downtrend
            elif price < s1 and volume_spike and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike2.0x_1wEMA50_v1"
timeframe = "4h"
leverage = 1.0