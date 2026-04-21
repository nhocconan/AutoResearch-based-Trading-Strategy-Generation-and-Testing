#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 Breakout + 12h EMA Trend + Volume Spike
# Long when price breaks above Camarilla R1 (resistance 1) and 12h EMA > 12h EMA 10 bars ago and 12h volume > 1.5x 20-bar average
# Short when price breaks below Camarilla S1 (support 1) and 12h EMA < 12h EMA 10 bars ago and 12h volume > 1.5x 20-bar average
# Exit when price crosses Camarilla pivot point (H+L+C)/3
# Uses 12h trend for direction and volume for confirmation to avoid false breakouts
# Target: 25-40 trades/year with tight conditions to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(20) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h EMA slope (trend direction)
    ema_slope = np.zeros_like(ema_12h)
    ema_slope[1:] = ema_12h[1:] - ema_12h[:-1]
    
    # Calculate 12h volume moving average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot point and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_12h_current = df_12h['volume'].iloc[i // 2] if i >= 2 else df_12h['volume'].iloc[0]
        vol_confirm = vol_12h_current > 1.5 * vol_ma_12h_aligned[i]
        
        # Trend filter: EMA slope direction
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            if vol_confirm:
                # Long: price breaks above R1 with uptrend
                if price > r1_aligned[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 with downtrend
                elif price < s1_aligned[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses pivot point
            exit_signal = False
            
            if position == 1:  # long position
                if price < pivot_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0