#!/usr/bin/env python3
# 6h_1d_Williams_VIX_Fix_Signal
# Hypothesis: Uses Williams VIX Fix (volatility spike detector) on daily timeframe to identify high-volatility regimes.
# Combines with 6-hour price action: long when price > VWAP and volatility spike, short when price < VWAP and volatility spike.
# VIX Fix helps identify panic selling or euphoria buying, providing edge in both trending and ranging markets.
# Target: 20-40 trades/year to minimize fee drag while capturing volatility-driven moves.

name = "6h_1d_Williams_VIX_Fix_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for VIX Fix calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 22:  # Need at least 22 days for lookback
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Williams VIX Fix ---
    # VIX Fix = (Highest Close in lookback - Low) / Highest Close in lookback * 100
    # We invert it to get a volatility spike signal: higher = more fear
    lookback = 22
    highest_close = pd.Series(df_1d['close'].values).rolling(window=lookback, min_periods=lookback).max().values
    vix_fix = (highest_close - df_1d['low'].values) / highest_close * 100
    vix_fix_ma = pd.Series(vix_fix).rolling(window=10, min_periods=10).mean().values  # Smooth the signal
    
    # Align daily VIX Fix to 6h
    vix_fix_aligned = align_htf_to_ltf(prices, df_1d, vix_fix_ma)
    
    # --- 6-hour VWAP for intraday trend ---
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).cumsum().values
    vwap_den = pd.Series(volume).cumsum().values
    vwap = vwap_num / vwap_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for VIX Fix calculation (22 + 10 for smoothing)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vix_fix_aligned[i]) or
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility spike detection: VIX Fix above its 50-period upper band
        # Using rolling mean + 2*std as dynamic threshold
        if i >= 50:  # Need enough data for volatility regime detection
            vix_slice = vix_fix_aligned[start_idx:i+1]
            if len(vix_slice) >= 20:
                vix_mean = np.nanmean(vix_slice[-20:])
                vix_std = np.nanstd(vix_slice[-20:])
                volatility_threshold = vix_mean + 2.0 * vix_std
                volatility_spike = vix_fix_aligned[i] > volatility_threshold
            else:
                volatility_spike = False
        else:
            volatility_spike = False
        
        if position == 0:
            # Long: volatility spike + price above VWAP (buying panic/dip)
            if volatility_spike and close[i] > vwap[i]:
                signals[i] = 0.25
                position = 1
            # Short: volatility spike + price below VWAP (selling euphoria/rally)
            elif volatility_spike and close[i] < vwap[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: volatility subsides OR price crosses below VWAP
                if (not volatility_spike) or (close[i] < vwap[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: volatility subsides OR price crosses above VWAP
                if (not volatility_spike) or (close[i] > vwap[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals