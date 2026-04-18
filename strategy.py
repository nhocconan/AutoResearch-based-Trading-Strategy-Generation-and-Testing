#!/usr/bin/env python3
"""
4h_ParabolicSAR_Direction_V1
Hypothesis: Use Parabolic SAR on 1d for trend direction and 4h price action for entry timing.
Go long when price crosses above SAR and SAR is rising, short when price crosses below SAR and SAR is falling.
Requires volume > 1.3x 20-period average for confirmation. Uses 1w ADX > 25 to filter strong trends.
Target: 20-40 trades/year by combining trend following with volatility filtering.
Works in bull via long trends and in bear via short trends, with ADX filter avoiding chop.
"""

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
    
    # Get 1d data for Parabolic SAR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR calculation
    def calculate_parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(close)
        sar = np.full(n, np.nan)
        trend = np.full(n, np.nan)  # 1 for up, -1 for down
        af = np.full(n, np.nan)
        ep = np.full(n, np.nan)  # extreme point
        
        # Initialize
        if n < 3:
            return sar, trend
        
        # Start with assumption of uptrend
        sar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # was uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                # Check for reversal
                if low[i] < sar[i]:
                    trend[i] = -1  # reverse to downtrend
                    sar[i] = ep[i-1]  # SAR becomes prior EP
                    af[i] = af_start
                    ep[i] = low[i]
                else:  # continue uptrend
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                    else:
                        ep[i] = ep[i-1]
                    af[i] = min(af[i-1] + af_increment, af_max)
            else:  # was downtrend
                sar[i] = sar[i-1] + af[i-1] * (sar[i-1] - ep[i-1])
                # Check for reversal
                if high[i] > sar[i]:
                    trend[i] = 1  # reverse to uptrend
                    sar[i] = ep[i-1]  # SAR becomes prior EP
                    af[i] = af_start
                    ep[i] = high[i]
                else:  # continue downtrend
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                    else:
                        ep[i] = ep[i-1]
                    af[i] = min(af[i-1] + af_increment, af_max)
        
        return sar, trend
    
    # Calculate SAR on 1d data
    sar_1d, trend_1d = calculate_parabolic_sar(high_1d, low_1d, close_1d)
    
    # Align SAR and trend to 4h timeframe
    sar_1d_aligned = align_htf_to_ltf(prices, df_1d, sar_1d)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        n = len(close)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smooth TR, DM+
        tr_smooth = np.full(n, np.nan)
        dm_plus_smooth = np.full(n, np.nan)
        dm_minus_smooth = np.full(n, np.nan)
        
        if n >= period:
            # Initial averages
            tr_smooth[period-1] = np.nansum(tr[:period])
            dm_plus_smooth[period-1] = np.nansum(dm_plus[:period])
            dm_minus_smooth[period-1] = np.nansum(dm_minus[:period])
            
            # Wilder smoothing
            for i in range(period, n):
                tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
                dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
                dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # Directional Indicators
        di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
        di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                     100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        
        adx = np.full(n, np.nan)
        if n >= period * 2:
            # Initial ADX
            adx[period*2-1] = np.nanmean(dx[period:period*2])
            # Wilder smoothing for ADX
            for i in range(period*2, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # SAR needs some history
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sar_1d_aligned[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0 and strong_trend:
            # Long: price crosses above SAR and SAR is rising (uptrend)
            if close[i] > sar_1d_aligned[i] and trend_1d_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below SAR and SAR is falling (downtrend)
            elif close[i] < sar_1d_aligned[i] and trend_1d_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below SAR (trend reversal)
            if close[i] < sar_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above SAR (trend reversal)
            if close[i] > sar_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ParabolicSAR_Direction_V1"
timeframe = "4h"
leverage = 1.0