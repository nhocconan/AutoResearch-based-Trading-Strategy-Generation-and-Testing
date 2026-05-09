#!/usr/bin/env python3
# 12h_VWAP_Reversal_1dTrend_Volume
# Hypothesis: Mean reversion at VWAP deviation in trending markets. Long when price deviates below VWAP with volume support in uptrend; short when price deviates above VWAP with volume support in downtrend. Uses 1d trend filter (EMA50) and volume confirmation to avoid false signals. Designed for 12h timeframe with target 15-35 trades/year.

name = "12h_VWAP_Reversal_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate VWAP for 12h period
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_volume = np.cumsum(volume)
    vwap = np.full_like(close, np.nan)
    valid_vol = cum_volume != 0
    vwap[valid_vol] = cum_pv[valid_vol] / cum_volume[valid_vol]
    
    # Calculate VWAP deviation as percentage
    vwap_dev = np.full_like(close, np.nan)
    valid_vwap = ~np.isnan(vwap)
    vwap_dev[valid_vwap] = (close[valid_vwap] - vwap[valid_vwap]) / vwap[valid_vwap]
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap_dev[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price below VWAP (mean reversion) AND volume confirmation AND uptrend (price > EMA50)
            if vwap_dev[i] < -0.008 and volume_ratio[i] > 1.5 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price above VWAP (mean reversion) AND volume confirmation AND downtrend (price < EMA50)
            elif vwap_dev[i] > 0.008 and volume_ratio[i] > 1.5 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to VWAP or trend turns bearish
            if vwap_dev[i] > -0.002 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to VWAP or trend turns bullish
            if vwap_dev[i] < 0.002 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals