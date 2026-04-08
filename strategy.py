# 4h_bollinger_breakout_1d_trend_volume_v1
# Hypothesis: Bollinger Band breakouts with 1d trend filter and volume confirmation
# Works in bull/bear: Breakouts capture momentum; 1d trend filter avoids counter-trend trades; volume reduces false signals
# Target: 20-40 trades/year, <160 total over 4 years to avoid fee drag
# Edge: Bollinger bands adapt to volatility, breakouts signal institutional interest

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (4h)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    
    # 1d trend: 50-period EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 4h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(std_dev[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches middle band or trend fails
            if close[i] <= sma[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches middle band or trend fails
            if close[i] >= sma[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50_4h[i]
            bearish = close[i] < ema_50_4h[i]
            
            # Long: price breaks above upper band + bullish trend + volume
            if (close[i] > upper_band[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band + bearish trend + volume
            elif (close[i] < lower_band[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals