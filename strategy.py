#!/usr/bin/env python3
# Hypothesis: 4h Bollinger Bands squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB after low volatility squeeze (BB width < 20th percentile)
# Short when price breaks below lower BB after squeeze with opposite trend condition
# Uses Bollinger Bands for volatility breakout, 1d EMA50 for trend filter, volume > 1.5x average
# Designed to capture explosive moves after low volatility periods in both bull and bear markets
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "4h_BollingerSqueeze_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    
    upper_bb = ma + (bb_std * std)
    lower_bb = ma - (bb_std * std)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile for squeeze detection
    # Squeeze when BB width is below 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20)
    squeeze_condition = bb_width < bb_width_percentile.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper BB after squeeze, 1d EMA50 uptrend, volume spike
            if (close[i] > upper_bb[i] and 
                squeeze_condition[i-1] and  # Was in squeeze previously
                ema50_1d_aligned[i] > ema50_1d_aligned[i-20] and  # 1d EMA up over ~1 month
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB after squeeze, 1d EMA50 downtrend, volume spike
            elif (close[i] < lower_bb[i] and 
                  squeeze_condition[i-1] and  # Was in squeeze previously
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-20] and  # 1d EMA down over ~1 month
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle band or volatility expands significantly
            if (close[i] < ma[i] or 
                bb_width[i] > bb_width_percentile.values[i] * 2.0):  # BB width doubled
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle band or volatility expands significantly
            if (close[i] > ma[i] or 
                bb_width[i] > bb_width_percentile.values[i] * 2.0):  # BB width doubled
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals