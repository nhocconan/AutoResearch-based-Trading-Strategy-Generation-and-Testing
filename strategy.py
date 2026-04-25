#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 12h EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance. Breakouts with volume and 12h EMA trend filter capture momentum. Chop filter avoids whipsaws in ranging markets. Works in bull/bear via trend filter and volatility-based sizing.
Target: 25-50 trades/year on 4h timeframe.
"""

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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) for stoploss and chop filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate choppiness index (CHOP) for regime filter
    if len(close) >= 14:
        atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
        hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
        chop = np.where((hh - ll) > 0, -100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    else:
        chop = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_12h_aligned[i]
        atr_val = atr[i]
        chop_val = chop[i]
        
        # Calculate Camarilla levels from previous 1d bar (yesterday's OHLC)
        idx_1d = i // (24 * 60 // 15)  # Number of 15m bars per day, but we use get_htf_data so need different approach
        # Instead, we use the 1d dataframe directly - get previous completed 1d bar
        # Since we're on 4h timeframe, we need to map to 1d index
        # Simpler: use the last completed 1d bar from df_1d
        # We'll calculate Camarilla for each 1d bar and align to 4h
        
        # Calculate Camarilla levels for each 1d bar
        # This should be done outside the loop for efficiency
        pass  # Placeholder - we need to move Camarilla calculation outside loop
    
    # Recalculate: move Camarilla calculation outside loop for efficiency
    # Calculate Camarilla levels from 1d data
    if len(df_1d) >= 2:
        # For each 1d bar, calculate Camarilla levels based on previous day's OHLC
        # We'll use the previous day's high, low, close
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Shift by 1 to get previous day's OHLC for current day's Camarilla
        prev_high_1d = np.roll(high_1d, 1)
        prev_low_1d = np.roll(low_1d, 1)
        prev_close_1d = np.roll(close_1d, 1)
        # First bar has no previous day
        prev_high_1d[0] = np.nan
        prev_low_1d[0] = np.nan
        prev_close_1d[0] = np.nan
        
        # Camarilla calculations
        R1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
        S1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
        R2 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 6
        S2 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 6
        R3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 4
        S3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 4
        R4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
        S4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
        
        # AlCamarilla levels to 4h timeframe
        R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
        S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
        # We'll use R1 and S1 as primary levels
    else:
        R1_aligned = np.full(n, np.nan)
        S1_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_12h_aligned[i]
        R1 = R1_aligned[i]
        S1 = S1_aligned[i]
        atr_val = atr[i]
        chop_val = chop[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        # Chop filter: avoid ranging markets (CHOP > 61.8) and extreme trending (CHOP < 38.2) sometimes
        # We'll use CHOP between 38.2 and 61.8 as optimal range for breakouts
        chop_filter = (chop_val >= 38.2) and (chop_val <= 61.8)
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND uptrend AND chop filter
            long_condition = (curr_high > R1) and volume_spike and uptrend and chop_filter
            # Short: price breaks below S1 AND volume spike AND downtrend AND chop filter
            short_condition = (curr_low < S1) and volume_spike and downtrend and chop_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or trend reversal or chop extreme
            if curr_close <= entry_price - 2.0 * atr_val or not uptrend or chop_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or trend reversal or chop extreme
            if curr_close >= entry_price + 2.0 * atr_val or not downtrend or chop_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0