# 4h_VWAP_Deviation_MeanReversion_1dTrend
# Mean reversion to VWAP with daily trend filter. Works in both bull and bear by trading pullbacks to VWAP in trending markets.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)
# VWAP deviation >2σ triggers mean reversion entries when price is away from VWAP but aligned with daily trend.
# Exit when price returns to VWAP or trend breaks. Uses 0.25 position size to manage drawdown.

#!/usr/bin/env python3
name = "4h_VWAP_Deviation_MeanReversion_1dTrend"
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
    
    # === 4h VWAP ===
    typical_price = (high + low + close) / 3.0
    cum_vol_tp = np.nancumsum(volume * typical_price)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_vol_tp, cum_vol, out=np.full_like(cum_vol_tp, np.nan), where=cum_vol!=0)
    
    # VWAP deviation (%)
    vwap_dev = (close - vwap) / vwap * 100.0
    
    # Rolling std of VWAP deviation (20 periods)
    vwap_dev_series = pd.Series(vwap_dev)
    vwap_dev_std = vwap_dev_series.rolling(window=20, min_periods=20).std().values
    
    # === Daily Trend (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure enough data for VWAP std and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_dev[i]) or 
            np.isnan(vwap_dev_std[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: VWAP deviation > 2 standard deviations
        dev_threshold = 2.0 * vwap_dev_std[i]
        
        if position == 0:
            # Long: Price significantly below VWAP (-2σ) but daily trend up
            if (vwap_dev[i] < -dev_threshold and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price significantly above VWAP (+2σ) but daily trend down
            elif (vwap_dev[i] > dev_threshold and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price returns to VWAP or trend breaks down
            if vwap_dev[i] > -0.5 * dev_threshold or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price returns to VWAP or trend breaks up
            if vwap_dev[i] < 0.5 * dev_threshold or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals