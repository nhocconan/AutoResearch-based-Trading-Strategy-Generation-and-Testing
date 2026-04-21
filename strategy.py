#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Daily Williams %R with 4h Volume Spike and Trend Filter
# Uses daily Williams %R (overbought/oversold) for mean-reversion signals
# Only trades when 4h EMA50 confirms trend direction to avoid counter-trend whipsaws
# Requires volume > 1.5x 20-period average for confirmation
# Target: 20-40 trades/year by combining oversold/overbought signals with trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on daily data
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # Align Williams %R to 4h timeframe (no extra delay needed for Williams %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h EMA50 for trend filter
    close_series = pd.Series(prices['close'])
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below EMA50
        uptrend = price > ema_50[i]
        downtrend = price < ema_50[i]
        
        if position == 0:
            if volume_confirm:
                # Long when Williams %R oversold (< -80) and in uptrend
                if williams_r_aligned[i] < -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short when Williams %R overbought (> -20) and in downtrend
                elif williams_r_aligned[i] > -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long when Williams %R reaches overbought (> -20) or trend changes
                if williams_r_aligned[i] > -20 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit short when Williams %R reaches oversold (< -80) or trend changes
                if williams_r_aligned[i] < -80 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_EMA50_Volume"
timeframe = "4h"
leverage = 1.0