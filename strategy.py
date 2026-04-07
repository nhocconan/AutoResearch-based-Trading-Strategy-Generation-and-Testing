#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams %R + 1d Trend Filter + Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions on 4h chart.
# Trades in direction of 1d EMA50 trend with volume confirmation to avoid false signals.
# Williams %R < -80 = oversold (long signal in uptrend), > -20 = overbought (short signal in downtrend).
# Uses daily trend filter to ensure trades align with higher timeframe momentum.
# Targets 20-50 trades/year with strict entry conditions to minimize fee drag.

name = "4h_williams_r_1d_trend_volume_v1"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period) on 4h
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period, n):  # Start after warmup for Williams %R
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold OR trend turns down
            if williams_r[i] > -50 or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought OR trend turns up
            if williams_r[i] < -50 or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Williams %R oversold + volume confirmation + uptrend
            if (williams_r[i] < -80 and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: Williams %R overbought + volume confirmation + downtrend
            elif (williams_r[i] > -20 and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals