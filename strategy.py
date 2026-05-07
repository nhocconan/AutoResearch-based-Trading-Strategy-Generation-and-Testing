#!/usr/bin/env python3
"""
4H_RSI_4060_MeanReversion_1D_Volume_Trend_Filter_v1
Hypothesis: Mean reversion on 4h RSI (40-60 band) with 1d trend filter and volume confirmation.
Long when RSI crosses above 40 and price is above 1d EMA50; short when RSI crosses below 60 and price is below 1d EMA50.
Volume confirmation: current volume > 1.3x 20-period average volume.
This strategy captures mean reversion in ranging markets while filtering for trend direction to avoid whipsaws in strong trends.
"""
name = "4H_RSI_4060_MeanReversion_1D_Volume_Trend_Filter_v1"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_1d_50 = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup for RSI and volume
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.3 days on 4h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: RSI crosses above 40 and price above 1d EMA50
            if (rsi_values[i] > 40 and rsi_values[i-1] <= 40 and 
                close[i] > ema_1d_50_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: RSI crosses below 60 and price below 1d EMA50
            elif (rsi_values[i] < 60 and rsi_values[i-1] >= 60 and 
                  close[i] < ema_1d_50_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to neutral zone (50) or trend reversal
            if position == 1 and (rsi_values[i] >= 50 or close[i] < ema_1d_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (rsi_values[i] <= 50 or close[i] > ema_1d_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals