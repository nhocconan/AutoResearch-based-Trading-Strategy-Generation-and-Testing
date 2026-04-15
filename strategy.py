#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for HTF context
    daily = get_htf_data(prices, '1d')
    weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    whigh = weekly['high'].values
    wlow = weekly['low'].values
    wdonch_high = pd.Series(whigh).rolling(window=20, min_periods=20).max().values
    wdonch_low = pd.Series(wlow).rolling(window=20, min_periods=20).min().values
    wdonch_high_aligned = align_htf_to_ltf(prices, weekly, wdonch_high)
    wdonch_low_aligned = align_htf_to_ltf(prices, weekly, wdonch_low)
    
    # Calculate daily RSI (14-period) for momentum filter
    delta = pd.Series(daily['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, daily, rsi_values)
    
    # Calculate 6-period EMA for trend filter
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Calculate volume spike: current volume > 2x 6-period average volume
    vol_ema6 = pd.Series(volume).ewm(span=6, adjust=False, min_periods=6).mean().values
    vol_spike = volume > (2.0 * vol_ema6)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(wdonch_high_aligned[i]) or np.isnan(wdonch_low_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(ema6[i])):
            continue
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian high
        # 2. Daily RSI > 50 (bullish momentum)
        # 3. Price above 6-period EMA (short-term trend)
        # 4. Volume spike (confirmation)
        if (close[i] > wdonch_high_aligned[i] and 
            rsi_aligned[i] > 50 and 
            close[i] > ema6[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short conditions:
        # 1. Price breaks below weekly Donchian low
        # 2. Daily RSI < 50 (bearish momentum)
        # 3. Price below 6-period EMA (short-term trend)
        # 4. Volume spike (confirmation)
        elif (close[i] < wdonch_low_aligned[i] and 
              rsi_aligned[i] < 50 and 
              close[i] < ema6[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite breakout
        elif (close[i] < wdonch_low_aligned[i] and signals[i-1] > 0) or \
             (close[i] > wdonch_high_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyDonchian_RSI_Volume_Filter"
timeframe = "6h"
leverage = 1.0