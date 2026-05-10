#!/usr/bin/env python3
# 12h_ParabolicSAR_VolumeTrend
# Hypothesis: 12-hour Parabolic SAR trend following with volume confirmation and daily EMA50 trend filter.
# Parabolic SAR captures trend direction and provides trailing stop. Volume confirmation ensures breakout strength.
# Daily EMA50 filters for higher timeframe trend to avoid counter-trend trades. Designed for 12h to achieve 12-37 trades/year.

name = "12h_ParabolicSAR_VolumeTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Parabolic SAR calculation
    def calculate_parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(high)
        sar = np.full(n, np.nan)
        trend = np.full(n, np.nan)  # 1 for uptrend, -1 for downtrend
        af = np.full(n, af_start)
        ep = np.full(n, np.nan)  # extreme point
        
        # Initialize
        if n < 2:
            return sar, trend
        
        sar[0] = low[0]
        trend[0] = 1
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                # Reverse if price falls below SAR
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]  # SAR becomes previous EP
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                # Reverse if price rises above SAR
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]  # SAR becomes previous EP
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return sar, trend
    
    sar, psar_trend = calculate_parabolic_sar(high, low)
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(sar[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: SAR indicates uptrend, price above SAR, above daily EMA50, strong volume
            if psar_trend[i] == 1 and close[i] > sar[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: SAR indicates downtrend, price below SAR, below daily EMA50, strong volume
            elif psar_trend[i] == -1 and close[i] < sar[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: SAR flips to downtrend or price drops below SAR
            if psar_trend[i] == -1 or close[i] < sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: SAR flips to uptrend or price rises above SAR
            if psar_trend[i] == 1 or close[i] > sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals