#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h price closes outside Bollinger Bands (20,2) with volume spike and 1d trend filter
    # Works in bull/bear: volatility expansion captures breakouts; Bollinger mean reversion provides exits
    # Uses Bollinger Bands for volatility-based breakout detection, volume for confirmation, 1d EMA for trend filter
    
    # Load 1d data once for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + bb_std * std_1d
    lower_bb = sma_1d - bb_std * std_1d
    
    # 1d EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Bollinger Bands and EMA to 4h
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_1d_50_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above upper BB with volume spike and 1d uptrend
            if close[i] > upper_bb_aligned[i] and vol_spike[i] and close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower BB with volume spike and 1d downtrend
            elif close[i] < lower_bb_aligned[i] and vol_spike[i] and close[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle Bollinger Band (SMA)
            if position == 1:
                if close[i] < sma_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Breakout_Volume_Spike_1dEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0