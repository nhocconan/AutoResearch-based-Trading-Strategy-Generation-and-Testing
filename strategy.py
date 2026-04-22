#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Williams %R + 1-day EMA trend + volume spike.
# Uses Williams %R(14) for overbought/oversold signals, EMA(50) for trend direction,
# and volume spike for confirmation. Designed to capture reversals in trending markets
# while avoiding chop. Targets 15-40 trades/year with disciplined risk control.
# Williams %R < -80 = oversold (long), > -20 = overbought (short) in trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-day data for Williams %R and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-day Williams %R
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 50-day EMA for trend
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = williams_r_aligned[i]
        ema = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) AND price above EMA (uptrend) AND volume spike
            if wr < -80 and price > ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) AND price below EMA (downtrend) AND volume spike
            elif wr > -20 and price < ema and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on Williams %R crossing above -50 (momentum fading) OR price below EMA (trend change)
                if wr > -50 or price < ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on Williams %R crossing below -50 (momentum fading) OR price above EMA (trend change)
                if wr < -50 or price > ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_EMA_Trend_Volume"
timeframe = "12h"
leverage = 1.0