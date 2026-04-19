#!/usr/bin/env python3
"""
1d_WaveTrend_Signal_With_Volume_Confirmation
Hypothesis: WaveTrend oscillator on 1d timeframe with volume confirmation and 1w trend filter
WaveTrend identifies overbought/oversold conditions with reduced lag
Volume confirmation ensures institutional participation
1w trend filter (EMA34) avoids counter-trend trades
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
Works in bull/bear via trend filter and mean-reversion logic
"""

name = "1d_WaveTrend_Signal_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

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
    
    # WaveTrend Oscillator (WT) - similar to double smoothed stochastic
    def wave_trend(high, low, close, channel_length=10, average_length=21):
        # Typical Price
        tp = (high + low + close) / 3.0
        
        # ESA: Exponential Smoothed Average of TP
        esa = np.full_like(tp, np.nan)
        alpha_esa = 2.0 / (channel_length + 1)
        for i in range(len(tp)):
            if i == 0:
                esa[i] = tp[i]
            elif not np.isnan(tp[i]) and not np.isnan(esa[i-1]):
                esa[i] = esa[i-1] + alpha_esa * (tp[i] - esa[i-1])
            else:
                esa[i] = np.nan
        
        # D: Deviation of TP from ESA
        d = np.full_like(tp, np.nan)
        for i in range(len(tp)):
            if not np.isnan(tp[i]) and not np.isnan(esa[i]):
                d[i] = abs(tp[i] - esa[i])
        
        # DE: Double Smoothed D
        de = np.full_like(tp, np.nan)
        alpha_de = 2.0 / (channel_length + 1)
        for i in range(len(d)):
            if i == 0:
                de[i] = d[i] if not np.isnan(d[i]) else 0
            elif not np.isnan(d[i]) and not np.isnan(de[i-1]):
                de[i] = de[i-1] + alpha_de * (d[i] - de[i-1])
            else:
                de[i] = np.nan
        
        # Avoid division by zero
        ci = np.full_like(tp, np.nan)
        mask = (de > 0) & ~np.isnan(de)
        ci[mask] = (tp[mask] - esa[mask]) / (0.015 * de[mask])
        
        # TCI: Trend Channel Index
        tci = np.full_like(tp, np.nan)
        alpha_tci = 2.0 / (average_length + 1)
        for i in range(len(ci)):
            if i == 0:
                tci[i] = ci[i] if not np.isnan(ci[i]) else 0
            elif not np.isnan(ci[i]) and not np.isnan(tci[i-1]):
                tci[i] = tci[i-1] + alpha_tci * (ci[i] - tci[i-1])
            else:
                tci[i] = np.nan
        
        # WT1 and WT2
        wt1 = tci
        wt2 = np.full_like(tp, np.nan)
        for i in range(len(wt1)):
            if i < 4:
                wt2[i] = np.nan
            else:
                wt2[i] = np.nanmean(wt1[i-3:i+1])  # 4-period average
        
        return wt1, wt2
    
    # 1d data for WaveTrend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate WaveTrend on 1d data
    wt1, wt2 = wave_trend(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Align WaveTrend to 1d timeframe (already aligned since we're using 1d data directly)
    wt1_1d = wt1
    wt2_1d = wt2
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close
    ema_34_1w = np.full_like(df_1w['close'].values, np.nan)
    alpha_ewma = 2.0 / (34 + 1)
    for i in range(len(df_1w)):
        if i == 0:
            ema_34_1w[i] = df_1w['close'].values[i]
        elif not np.isnan(df_1w['close'].values[i]) and not np.isnan(ema_34_1w[i-1]):
            ema_34_1w[i] = ema_34_1w[i-1] + alpha_ewma * (df_1w['close'].values[i] - ema_34_1w[i-1])
        else:
            ema_34_1w[i] = np.nan
    
    # Align EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wt1_1d[i]) or np.isnan(wt2_1d[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade long when price > weekly EMA34, short when price < weekly EMA34
        # This ensures we trade with the higher timeframe trend
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: WT1 crosses above WT2 from oversold (WT1 < -60) with volume and uptrend
            if (wt1[i] > wt2[i] and wt1[i-1] <= wt2[i-1] and 
                wt1[i] < -60 and  # Oversold condition
                volume_confirm[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: WT1 crosses below WT2 from overbought (WT1 > 60) with volume and downtrend
            elif (wt1[i] < wt2[i] and wt1[i-1] >= wt2[i-1] and 
                  wt1[i] > 60 and   # Overbought condition
                  volume_confirm[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if WT1 crosses below WT2 or trend changes
            if (wt1[i] < wt2[i] and wt1[i-1] >= wt2[i-1]) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if WT1 crosses above WT2 or trend changes
            if (wt1[i] > wt2[i] and wt1[i-1] <= wt2[i-1]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals