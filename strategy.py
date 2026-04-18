#!/usr/bin/env python3
"""
12h_ParabolicSAR_Trend_With_Volume_And_Trend_Filter
Hypothesis: Parabolic SAR signals combined with volume spike and 1d EMA50 trend filter.
Works in both bull and bear markets by following trend with SAR and confirming with volume and higher timeframe trend.
Target: 15-30 trades/year to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parabolic SAR calculation
    def calculate_parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)
        ep = np.zeros(n)  # extreme point
        
        # Initialize
        if high[1] > high[0]:
            trend[0] = 1
            sar[0] = low[0]
            ep[0] = high[0]
        else:
            trend[0] = -1
            sar[0] = high[0]
            ep[0] = low[0]
        af[0] = af_start
        
        for i in range(1, n):
            # SAR calculation
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Determine trend and EP
            if trend[i-1] == 1:  # uptrend
                if low[i] <= sar[i]:  # trend reversal
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:  # continue uptrend
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # downtrend
                if high[i] >= sar[i]:  # trend reversal
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:  # continue downtrend
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return sar, trend
    
    # Calculate SAR and trend
    sar, psar_trend = calculate_parabolic_sar(high, low)
    
    # Daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: >2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(sar[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sar_val = sar[i]
        psar_trend_val = psar_trend[i]
        ema50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: SAR bullish (price above SAR) with volume spike and uptrend on 1d
            if price > sar_val and psar_trend_val == 1 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: SAR bearish (price below SAR) with volume spike and downtrend on 1d
            elif price < sar_val and psar_trend_val == -1 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below SAR OR trend turns down on 1d
            if price < sar_val:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above SAR OR trend turns up on 1d
            if price > sar_val:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_ParabolicSAR_Trend_With_Volume_And_Trend_Filter"
timeframe = "12h"
leverage = 1.0