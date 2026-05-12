#!/usr/bin/env python3
name = "4h_Parabolic_SAR_Trend_Follow"
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
    
    # Parabolic SAR calculation (Wilder's)
    def calculate_parabolic_sar(high, low, start=0.02, increment=0.02, maximum=0.2):
        n = len(high)
        sar = np.zeros(n)
        trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
        af = np.full(n, start)
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        ep = high[0]  # extreme point
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep - sar[i-1])
                if sar[i] > low[i]:
                    # trend reversal
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af[i] = start
                else:
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                    af[i] = min(af[i-1] + increment, maximum)
            else:  # downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep - sar[i-1])
                if sar[i] < high[i]:
                    # trend reversal
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af[i] = start
                else:
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                    af[i] = min(af[i-1] + increment, maximum)
        return sar, trend
    
    # Calculate SAR on price data
    sar, psar_trend = calculate_parabolic_sar(high, low)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sar[i]) or np.isnan(psar_trend[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above SAR + above 1d EMA50 + volume filter
            if close[i] > sar[i] and close[i] > ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below SAR + below 1d EMA50 + volume filter
            elif close[i] < sar[i] and close[i] < ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below SAR
            if close[i] < sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above SAR
            if close[i] > sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals