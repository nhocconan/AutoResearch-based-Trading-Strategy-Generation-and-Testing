#!/usr/bin/env python3
name = "6h_ParabolicSAR_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data ONCE for trend and SAR
    df_1d = get_htf_data(prices, '1d')
    
    # Parabolic SAR calculation (AF=0.02, max=0.2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize SAR array
    sar = np.zeros_like(close_1d)
    trend = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    af = 0.02
    max_af = 0.2
    
    # Set initial values
    sar[0] = low_1d[0]
    ep = high_1d[0]  # Extreme point
    trend[0] = 1
    
    for i in range(1, len(close_1d)):
        if trend[i-1] == 1:  # Uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low_1d[i] < sar[i]:  # Trend reversal
                trend[i] = -1
                sar[i] = ep
                ep = low_1d[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
                else:
                    af = min(af + 0.02, max_af)
        else:  # Downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high_1d[i] > sar[i]:  # Trend reversal
                trend[i] = 1
                sar[i] = ep
                ep = high_1d[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
                else:
                    af = min(af + 0.02, max_af)
    
    # Align SAR and trend to 6h
    sar_aligned = align_htf_to_ltf(prices, df_1d, sar)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Volume filter: 24-period EMA for spike detection (4 days on 6h chart)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 1.5
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(sar_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_sar = close[i] > sar_aligned[i]
        price_below_sar = close[i] < sar_aligned[i]
        uptrend = trend_aligned[i] == 1
        downtrend = trend_aligned[i] == -1
        
        if position == 0:
            # Long: Price above SAR + uptrend + volume spike
            if price_above_sar and uptrend and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price below SAR + downtrend + volume spike
            elif price_below_sar and downtrend and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below SAR OR trend reverses to downtrend
                if price_below_sar or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above SAR OR trend reverses to uptrend
                if price_above_sar or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals