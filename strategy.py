#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for monthly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate monthly high/low from daily data (using 20-day rolling window for monthly)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Monthly high/low (20-day lookback)
    monthly_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    monthly_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate monthly pivot points (using monthly high/low/close)
    monthly_close = pd.Series(close_1d).rolling(window=20, min_periods=20).last().values
    monthly_pivot = (monthly_high + monthly_low + monthly_close) / 3.0
    monthly_r1 = 2 * monthly_pivot - monthly_low
    monthly_s1 = 2 * monthly_pivot - monthly_high
    
    # Align monthly pivot levels to 4h timeframe
    monthly_pivot_aligned = align_htf_to_ltf(prices, df_1d, monthly_pivot)
    monthly_r1_aligned = align_htf_to_ltf(prices, df_1d, monthly_r1)
    monthly_s1_aligned = align_htf_to_ltf(prices, df_1d, monthly_s1)
    
    # 50-period moving average for trend filter (4h timeframe)
    close = prices['close'].values
    ma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(monthly_pivot_aligned[i]) or 
            np.isnan(monthly_r1_aligned[i]) or 
            np.isnan(monthly_s1_aligned[i]) or 
            np.isnan(ma_50[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot = monthly_pivot_aligned[i]
        r1 = monthly_r1_aligned[i]
        s1 = monthly_s1_aligned[i]
        ma = ma_50[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below 50-period MA
        uptrend = price > ma
        downtrend = price < ma
        
        if position == 0:
            # Long: Price crosses above monthly R1 + uptrend + volume confirmation
            if price > r1 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below monthly S1 + downtrend + volume confirmation
            elif price < s1 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Price crosses back below/above monthly pivot or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below monthly pivot or trend reversal
                if price < pivot or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above monthly pivot or trend reversal
                if price > pivot or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_MonthlyPivot_R1_S1_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0