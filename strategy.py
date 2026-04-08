#!/usr/bin/env python3
# 1d_cci_trend_filter_volume_v1
# Hypothesis: Daily CCI (20) with trend filter and volume confirmation.
# Long: CCI crosses above -100 in uptrend (price > 200-day SMA) with volume > 1.5x average.
# Short: CCI crosses below +100 in downtrend (price < 200-day SMA) with volume > 1.5x average.
# Uses weekly trend filter to avoid counter-trend trades. Targets 15-25 trades/year.
# Works in bull markets by catching pullbacks in uptrends and in bear markets by
# catching bounces in downtrends. Volume filter reduces false signals.

name = "1d_cci_trend_filter_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # CCI calculation (20-period)
    cci_period = 20
    typical_price = (high + low + close) / 3.0
    tp_mean = np.zeros_like(typical_price)
    tp_mean[cci_period-1:] = np.convolve(typical_price, np.ones(cci_period)/cci_period, mode='valid')
    tp_mean[:cci_period-1] = tp_mean[cci_period-1]
    
    tp_std = np.zeros_like(typical_price)
    for i in range(cci_period-1, len(typical_price)):
        tp_std[i] = np.std(typical_price[i-cci_period+1:i+1])
    tp_std[:cci_period-1] = tp_std[cci_period-1]
    
    # Avoid division by zero
    tp_std[tp_std == 0] = 1e-10
    
    cci = (typical_price - tp_mean) / (0.015 * tp_std)
    
    # Weekly trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly SMA (50-period) for trend filter
    sma_period = 50
    sma_weekly = np.zeros_like(close_weekly)
    sma_weekly[sma_period-1:] = np.convolve(close_weekly, np.ones(sma_period)/sma_period, mode='valid')
    sma_weekly[:sma_period-1] = sma_weekly[sma_period-1]
    
    # Align weekly SMA to daily timeframe
    sma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma_weekly)
    
    # Volume filter: 20-day average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]
    
    # Pre-compute signals
    cci_cross_up = (cci > -100) & (np.roll(cci, 1) <= -100)  # Cross above -100
    cci_cross_down = (cci < 100) & (np.roll(cci, 1) >= 100)  # Cross below +100
    
    # Trend filters
    uptrend = close > sma_weekly_aligned
    downtrend = close < sma_weekly_aligned
    
    # Volume confirmation
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(cci_period, sma_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(sma_weekly_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit if CCI crosses below +100 or trend fails
            if cci[i] < 100 or not uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if CCI crosses above -100 or trend fails
            if cci[i] > -100 or not downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100, volume confirmation, and uptrend
            if cci_cross_up[i] and volume_filter[i] and uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below +100, volume confirmation, and downtrend
            elif cci_cross_down[i] and volume_filter[i] and downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals