#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 1d Trend + Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold extremes on 6h timeframe.
# Trades are taken only in the direction of 1d EMA50 trend to work in both bull and bear markets.
# Volume confirmation ensures moves have institutional participation.
# Targets 15-35 trades/year with disciplined entries to avoid overtrading.

name = "6h_williams_r_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values range from -100 to 0, with -20 to 0 = overbought, -80 to -100 = oversold
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, (highest_high - close) / rr * -100, -50)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period, n):  # Start after warmup for Williams %R and volume SMA
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold territory OR trend turns down
            if williams_r[i] > -20 or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought territory OR trend turns up
            if williams_r[i] < -80 or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Williams %R oversold (< -80) + volume confirmation + uptrend
            if (williams_r[i] < -80 and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R overbought (> -20) + volume confirmation + downtrend
            elif (williams_r[i] > -20 and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals