#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA200 Trend Filter + Volume Spike
# Long when Williams %R < -80 (oversold) and price > 1d EMA200 and volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) and price < 1d EMA200 and volume > 1.5x 20-period average
# Exit when Williams %R crosses -50 (mean reversion midpoint)
# Williams %R identifies reversal points in both bull and bear markets
# EMA200 filter ensures we trade with the higher timeframe trend
# Volume spike confirms momentum behind the move
# Target: 20-35 trades/year by requiring strict oversold/overbought conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Williams %R(14) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest High and Lowest Low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when HH == LL
    williams_r[highest_high == lowest_low] = -50  # neutral value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = df_1d['volume'].iloc[i // 96] > 1.5 * vol_ma if i >= 96 else df_1d['volume'].iloc[0] > 1.5 * vol_ma
        
        if position == 0:
            # Long: oversold, above EMA200, volume spike
            if williams_r[i] < -80 and price > ema200_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: overbought, below EMA200, volume spike
            elif williams_r[i] > -20 and price < ema200_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Williams %R crosses -50 (mean reversion)
            exit_signal = False
            
            if position == 1:  # long position
                if williams_r[i] > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA200_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0