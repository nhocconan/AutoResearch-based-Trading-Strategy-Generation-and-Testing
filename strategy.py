#!/usr/bin/env python3
"""
12h_1d_VWAP_Ratio_Deviation_MeanReversion
Hypothesis: Price deviates from volume-weighted average price (VWAP) during low volatility periods, then mean-reverts. Uses 1-day VWAP as fair value reference, enters when price deviates >1.5σ from VWAP with low volatility (ATR < 20-period ATR mean). Exits when price returns to VWAP. Works in bull/bear by fading extreme moves regardless of trend direction. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from math import sqrt
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP: cumulative(volume * price) / cumulative(volume)
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_numerator = typical_price * df_1d['volume'].values
    vwap_denominator = df_1d['volume'].values
    
    # Cumulative sums for VWAP
    cum_vwap_num = np.cumsum(vwap_numerator)
    cum_vwap_den = np.cumsum(vwap_denominator)
    vwap_1d = np.divide(cum_vwap_num, cum_vwap_den, 
                        out=np.full_like(cum_vwap_num, np.nan), 
                        where=cum_vwap_den!=0)
    
    # Align daily VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate price deviation from VWAP
    price_dev = close - vwap_aligned
    
    # Calculate rolling standard deviation of price deviation (20-period)
    dev_mean = np.full_like(price_dev, np.nan)
    dev_std = np.full_like(price_dev, np.nan)
    lookback = 20
    
    if len(price_dev) >= lookback:
        for i in range(lookback, len(price_dev)):
            window = price_dev[i-lookback:i]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                dev_mean[i] = np.mean(valid)
                dev_std[i] = np.std(valid)
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.full_like(tr, np.nan)
    atr_period = 20
    if len(tr) >= atr_period:
        for i in range(atr_period, len(tr)):
            atr[i] = np.mean(tr[i-atr_period:i])
    
    # ATR mean for volatility regime filter
    atr_mean = np.full_like(atr, np.nan)
    atr_lookback = 20
    if len(atr) >= atr_lookback:
        for i in range(atr_lookback, len(atr)):
            window = atr[i-atr_lookback:i]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                atr_mean[i] = np.mean(valid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, lookback, atr_period, atr_lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(dev_std[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_mean[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is low (ATR < ATR mean)
        low_vol = atr[i] < atr_mean[i]
        
        # Deviation threshold: 1.5 standard deviations
        upper_threshold = dev_mean[i] + 1.5 * dev_std[i]
        lower_threshold = dev_mean[i] - 1.5 * dev_std[i]
        
        if position == 0:
            # Long: price deviates below VWAP (oversold) with low volatility
            if price_dev[i] < lower_threshold and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: price deviates above VWAP (overbought) with low volatility
            elif price_dev[i] > upper_threshold and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP (deviation crosses zero)
            if price_dev[i] > dev_mean[i]:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP (deviation crosses zero)
            if price_dev[i] < dev_mean[i]:  # Return to mean
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_VWAP_Ratio_Deviation_MeanReversion"
timeframe = "12h"
leverage = 1.0