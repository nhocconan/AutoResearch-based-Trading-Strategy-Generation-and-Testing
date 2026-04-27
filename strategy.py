#!/usr/bin/env python3
"""
#100760 - 4h_RSI_RSI_Divergence_1dTrend_VolumeFilter
Hypothesis: RSI divergence with 1d trend filter and volume confirmation. Works in bull (bullish divergence + uptrend) and bear (bearish divergence + downtrend). Uses 4h RSI(14) for divergence detection, 1d EMA50 for trend, and volume spike for confirmation. Targets 20-40 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Find RSI divergence: bullish (price low, RSI higher low) and bearish (price high, RSI lower high)
    # We'll use a simple approach: look for RSI turning points with price confirmation
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for RSI oversold/overbought conditions as potential reversal points
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: volume > 1.3x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume[i] > (vol_ma[i] * 1.3) if not np.isnan(vol_ma[i]) else False
        
        # Bullish setup: RSI oversold, price above 1d EMA50 (uptrend), volume spike
        if (rsi[i] < 30 and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter):
            signals[i] = 0.25
            position = 1
        # Bearish setup: RSI overbought, price below 1d EMA50 (downtrend), volume spike
        elif (rsi[i] > 70 and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter):
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone (50) or opposite extreme
        elif position == 1 and (rsi[i] >= 50 or rsi[i] > 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi[i] <= 50 or rsi[i] < 30):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_RSI_Divergence_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0