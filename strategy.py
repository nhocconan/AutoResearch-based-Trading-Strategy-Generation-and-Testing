#!/usr/bin/env python3
# 1d_WeeklyVWAP_MeanReversion
# Hypothesis: Mean reversion toward weekly VWAP on daily timeframe. Long when price is significantly below weekly VWAP with volume confirmation, short when significantly above. Weekly VWAP acts as dynamic support/resistance, effective in both bull and bear markets as price tends to revert to weekly average. Volume filter ensures mean reversion attempts have participation. Target: 10-25 trades/year per symbol with low turnover.

name = "1d_WeeklyVWAP_MeanReversion"
timeframe = "1d"
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
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for each weekly bar: cumulative (price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pv = typical_price * df_1w['volume']
    cum_pv = np.cumsum(pv.values)
    cum_vol = np.cumsum(df_1w['volume'].values)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)
    
    # Align weekly VWAP to daily timeframe (no additional delay needed as VWAP is cumulative)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # 20-day standard deviation of price deviation from VWAP for dynamic threshold
    price_dev = close - vwap_aligned
    # Calculate rolling std dev of price deviation
    dev_ma = np.full_like(price_dev, np.nan)
    dev_ma_sq = np.full_like(price_dev, np.nan)
    
    if len(price_dev) >= 20:
        # Initialize first 20 values
        dev_ma[19] = np.nanmean(price_dev[0:20])
        dev_ma_sq[19] = np.nanmean(price_dev[0:20] ** 2)
        for i in range(20, len(price_dev)):
            if not np.isnan(price_dev[i]):
                dev_ma[i] = 0.95 * dev_ma[i-1] + 0.05 * price_dev[i]
                dev_ma_sq[i] = 0.95 * dev_ma_sq[i-1] + 0.05 * (price_dev[i] ** 2)
            else:
                dev_ma[i] = dev_ma[i-1]
                dev_ma_sq[i] = dev_ma_sq[i-1]
    
    # Calculate standard deviation: sqrt(E[X^2] - E[X]^2)
    dev_var = dev_ma_sq - (dev_ma ** 2)
    dev_std = np.sqrt(np.maximum(dev_var, 0))
    
    # Dynamic threshold: 2 standard deviations
    threshold = 2.0 * dev_std
    
    # Volume filter: current volume vs 20-day average
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
    
    start_idx = max(20, 20)  # Need deviation stats and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(threshold[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dev = price_dev[i]
        thresh = threshold[i]
        vol_ratio = volume_ratio[i]
        
        if position == 0:
            # Enter long: price below VWAP by more than threshold with volume confirmation
            if dev < -thresh and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: price above VWAP by more than threshold with volume confirmation
            elif dev > thresh and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP (mean reversion complete) or volatility too low
            if dev > -0.5 * thresh:  # Exit when halfway back to VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP
            if dev < 0.5 * thresh:  # Exit when halfway back to VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals