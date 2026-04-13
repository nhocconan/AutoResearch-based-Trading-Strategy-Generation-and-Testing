#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme with 1w EMA200 trend filter and volume confirmation
    # Long: Williams %R(14) < -80 (oversold) AND price > weekly EMA200 (bullish trend) AND volume > 1.5x avg
    # Short: Williams %R(14) > -20 (overbought) AND price < weekly EMA200 (bearish trend) AND volume > 1.5x avg
    # Exit: Williams %R returns to -50 level (mean reversion) OR opposite extreme
    # Using 6h timeframe for optimal trade frequency (target 12-37/year), Williams %R for momentum extremes,
    # weekly EMA200 for major trend filter, and volume confirmation to avoid false signals.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200
    close_1w = df_1w['close'].values
    ema_200 = np.full_like(close_1w, np.nan)
    multiplier = 2.0 / (200 + 1)
    ema_200[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_200[i] = (close_1w[i] * multiplier) + (ema_200[i-1] * (1 - multiplier))
    
    # Align weekly EMA200 to 6h
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate 6h Williams %R(14)
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA200
        bullish_trend = close[i] > ema_200_aligned[i]
        bearish_trend = close[i] < ema_200_aligned[i]
        
        # Williams %R extreme conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        mean_reversion = abs(williams_r[i] + 50) < 10  # near -50 level
        
        # Entry logic: Williams %R extreme + trend filter + volume confirmation
        long_entry = oversold and bullish_trend and volume_spike[i]
        short_entry = overbought and bearish_trend and volume_spike[i]
        
        # Exit logic: mean reversion or opposite extreme
        long_exit = mean_reversion or overbought
        short_exit = mean_reversion or oversold
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_williamsr_extreme_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0