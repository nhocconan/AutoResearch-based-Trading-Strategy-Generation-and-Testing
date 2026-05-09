#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ParabolicSAR_EMA200_Volume_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate Parabolic SAR on 6h data
    sar = np.full(n, np.nan)
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02
    max_af = 0.2
    ep = high[0]  # Extreme point
    
    # Initialize SAR for first bar
    sar[0] = low[0]
    
    for i in range(1, n):
        if trend == 1:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low[i] < sar[i]:  # Trend reversal
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high[i] > sar[i]:  # Trend reversal
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Calculate EMA200 from 1d data
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 200)  # Need enough data for volume MA and EMA200
    
    for i in range(start_idx, n):
        if np.isnan(sar[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        psar = sar[i]
        ema200 = ema200_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price above SAR (uptrend) and above EMA200, with volume
            if close[i] > psar and close[i] > ema200 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below SAR (downtrend) and below EMA200, with volume
            elif close[i] < psar and close[i] < ema200 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below SAR (trend reversal)
            if close[i] < psar:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above SAR (trend reversal)
            if close[i] > psar:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals