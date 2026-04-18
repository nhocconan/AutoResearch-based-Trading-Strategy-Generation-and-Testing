#!/usr/bin/env python3
"""
4h Volatility Contraction Pattern with Volume Expansion and 1d Trend Filter
Hypothesis: Low volatility contractions (narrow Bollinger Bands) followed by
expansion with volume capture explosive moves. Works in both bull and bear
markets by only taking breakouts aligned with 1d EMA trend. Low trade frequency
(~25/year) minimizes fee drag while capturing strong directional moves.
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Bollinger Bands (20, 2.0) for volatility contraction/expansion
    bb_period = 20
    bb_mult = 2.0
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = ma + bb_mult * std
    lower = ma - bb_mult * std
    bandwidth = (upper - lower) / ma  # Bandwidth as % of mean
    
    # Bollinger Band squeeze detection: bandwidth < 20th percentile
    bandwidth_series = pd.Series(bandwidth)
    bw_percentile = bandwidth_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bandwidth < bw_percentile
    
    # Volume filter: current volume > 2.0x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_expansion = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ma[i]) or 
            np.isnan(std[i]) or np.isnan(bw_percentile[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema34_1d_aligned[i]
        bw = bandwidth[i]
        bw_pctl = bw_percentile[i]
        vol_ok = vol_expansion[i]
        sqz = squeeze[i]
        
        if position == 0:
            # Look for volatility expansion after squeeze, with volume, in trend direction
            if not sqz and bw > bw_pctl and vol_ok:
                # Breakout above upper band with volume in uptrend
                if price > upper[i] and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Breakout below lower band with volume in downtrend
                elif price < lower[i] and price < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to middle band or volatility contracts again
            if price < ma[i] or (not sqz and bw < bw_pctl):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to middle band or volatility contracts again
            if price > ma[i] or (not sqz and bw < bw_pctl):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volatility_Contraction_Expansion_Volume_Trend"
timeframe = "4h"
leverage = 1.0