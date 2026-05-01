#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# (-20 for overbought, -80 for oversold) with volume spike indicate high-probability
# mean-reversion trades. 1d EMA50 ensures alignment with daily trend to avoid
# counter-trend trades in strong markets. Designed for 6h timeframe to balance
# trade frequency and signal quality, targeting 12-37 trades/year.

name = "6h_WilliamsR_Extreme_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R calculation (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 is overbought, < -80 is oversold
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, lookback)  # Need sufficient history for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Williams %R extreme levels
        overbought = williams_r[i] > -20
        oversold = williams_r[i] < -80
        
        # Williams %R reversal signals: exiting extreme territory
        # Long signal: was oversold, now rising above -80 with volume spike in uptrend
        # Short signal: was overbought, now falling below -20 with volume spike in downtrend
        if position == 0:  # Flat - look for new entries
            # Long: reversal from oversold, volume spike, uptrend
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: reversal from overbought, volume spike, downtrend
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on reversal from overbought or trend reversal
            if williams_r[i] < -20 and williams_r[i-1] >= -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on reversal from oversold or trend reversal
            if williams_r[i] > -80 and williams_r[i-1] <= -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals