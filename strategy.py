#!/usr/bin/env python3
"""
4h_parabolic_sar_12h_adx_volume_v1
Hypothesis: Parabolic SAR on 4h captures trend direction with built-in acceleration, while 12h ADX > 25 filters for strong trends and volume confirmation ensures institutional participation. This combination works in both bull (trend following) and bear (short trends) markets. Target: 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_parabolic_sar_12h_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parabolic SAR on 4h
    def calculate_psar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(high)
        psar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)
        ep = np.zeros(n)  # extreme point
        
        # Initialize
        psar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                # Ensure PSAR doesn't exceed prior two lows
                if i >= 2:
                    psar[i] = min(psar[i], low[i-1], low[i-2])
                
                # Trend reversal
                if low[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = ep[i-1]  # SAR becomes prior EP
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # downtrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                # Ensure PSAR doesn't fall below prior two highs
                if i >= 2:
                    psar[i] = max(psar[i], high[i-1], high[i-2])
                
                # Trend reversal
                if high[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = ep[i-1]  # SAR becomes prior EP
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return psar, trend
    
    psar, psar_trend = calculate_psar(high, low)
    
    # 12h ADX for trend strength
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.max([high_12h[0] - low_12h[0], 
                                   np.abs(high_12h[0] - close_12h[0]),
                                   np.abs(low_12h[0] - close_12h[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.sum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_12h = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h > 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h > 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = smooth_series(dx, 14)
    
    # Align 12h ADX to 4h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume filter: 20-period average on 4h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for indicators
        # Skip if data not available
        if (np.isnan(psar[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        # ADX filter: trend strength > 25
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: PSAR flip to downtrend
            if psar_trend[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: PSAR flip to uptrend
            if psar_trend[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and strong_trend:
                # Long entry: PSAR uptrend
                if psar_trend[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short entry: PSAR downtrend
                elif psar_trend[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals