#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R from 1d + volume spike + volume imbalance (taker buy ratio)
# Long when %R < -80 (oversold) with volume spike and buyer dominance
# Short when %R > -20 (overbought) with volume spike and seller dominance
# Williams %R: -100 * (Highest High - Close) / (Highest High - Lowest Low) over 14 periods
# Volume spike: current volume > 1.5x 20-period average
# Volume imbalance: taker_buy_ratio > 0.6 for long, < 0.4 for short
# Target: 15-30 trades/year by requiring oversold/overbought + volume confirmation
# Works in bull/bear: Williams %R identifies extremes, volume filters avoid fakeouts

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula
    wr = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(len(close_1d)):
        if highest_high[i] > lowest_low[i]:  # avoid division by zero
            wr[i] = -100 * (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i])
    
    # Align Williams %R to 6h timeframe (no extra delay needed for Williams %R)
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Taker buy ratio (already in data)
    taker_buy_ratio = (prices['taker_buy_volume'] / prices['volume']).values
    # Handle division by zero
    taker_buy_ratio = np.where(prices['volume'] == 0, 0.5, taker_buy_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(wr_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(taker_buy_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Volume imbalance confirmation
        buy_pressure = taker_buy_ratio[i] > 0.6
        sell_pressure = taker_buy_ratio[i] < 0.4
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R oversold (< -80) with buyer dominance
                if wr_aligned[i] < -80 and buy_pressure:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought (> -20) with seller dominance
                elif wr_aligned[i] > -20 and sell_pressure:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R returns from oversold (> -50) or loses buyer momentum
                if wr_aligned[i] > -50 or taker_buy_ratio[i] < 0.5:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R returns from overbought (< -50) or loses seller momentum
                if wr_aligned[i] < -50 or taker_buy_ratio[i] > 0.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR14_1dVolumeImbalance_VolumeSpike"
timeframe = "6h"
leverage = 1.0