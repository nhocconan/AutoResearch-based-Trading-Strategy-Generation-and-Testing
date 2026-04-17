#!/usr/bin/env python3
"""
6h_ParabolicSAR_Trend_Continuation
Hypothesis: On 6h timeframe, enter long when Parabolic SAR flips below price with daily volume confirmation and bullish daily trend; short when SAR flips above price with volume and bearish daily trend. Parabolic SAR captures trend reversals effectively in trending markets, while volume and daily trend filters avoid whipsaws in ranging conditions. Designed for 50-150 total trades over 4 years to minimize fee drag and work in both bull/bear markets via trend alignment.
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
    
    # === Daily data for volume and trend ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Parabolic SAR on 6h data
    # Parameters: start=0.02, increment=0.02, maximum=0.2
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    
    # Initialize
    sar[0] = low[0]
    trend[0] = 1
    ep = high[0]  # extreme point
    
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            # SAR cannot exceed the low of the past two periods
            sar[i] = min(sar[i], low[i-1], low[i-2] if i>=2 else low[i-1])
            if low[i] < sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep  # SAR becomes the prior EP
                ep = low[i]  # reset EP to current low
                af = 0.02
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            # SAR cannot be below the high of the past two periods
            sar[i] = max(sar[i], high[i-1], high[i-2] if i>=2 else high[i-1])
            if high[i] > sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep  # SAR becomes the prior EP
                ep = high[i]  # reset EP to current high
                af = 0.02
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Daily 20-period average volume for confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # Daily 50-period SMA for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers Parabolic SAR initialization and daily indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(vol_avg20_1d_aligned[i]) or np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Trend filter: price above/below daily 50 SMA
        above_trend = close[i] > sma50_1d_aligned[i]
        below_trend = close[i] < sma50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: SAR below price + volume + above daily trend
            if sar[i] < close[i] and vol_filter and above_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: SAR above price + volume + below daily trend
            elif sar[i] > close[i] and vol_filter and below_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal when SAR flips
        elif position == 1:
            if sar[i] > close[i]:  # SAR flips above price = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if sar[i] < close[i]:  # SAR flips below price = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ParabolicSAR_Trend_Continuation"
timeframe = "6h"
leverage = 1.0