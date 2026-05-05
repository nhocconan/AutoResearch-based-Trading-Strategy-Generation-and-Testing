#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume confirmation and 1d trend filter
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND 12h volume > 1.5x 20-period average AND close > 1d EMA50
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile AND 12h volume > 1.5x 20-period average AND close < 1d EMA50
# Exit when price crosses 20-period SMA (mean reversion to intermediate trend)
# Uses Bollinger squeeze to identify low volatility periods primed for expansion
# Volume confirmation from 12h ensures institutional participation
# 1d EMA50 filter avoids counter-trend trades in strong trends
# Discrete sizing (0.25) limits fee drag and manages drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_BB_Squeeze_Breakout_12hVolume_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
        # Calculate 20th percentile of BB width for squeeze detection
        bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
        squeeze_filter = bb_width < bb_width_percentile
    else:
        sma_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        squeeze_filter = np.zeros(n, dtype=bool)
    
    # Calculate volume spike filter on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
        volume_filter = vol_12h >= (1.5 * vol_ma_20_12h_aligned)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) if isinstance(volume_filter[i], np.floating) else False):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Handle volume_filter which might be boolean or float
        vol_ok = volume_filter[i] if isinstance(volume_filter[i], (bool, np.bool_)) else (volume_filter[i] > 0.5)
        
        if position == 0:
            # Long conditions: price breaks above upper BB AND squeeze AND 12h volume spike AND above 1d EMA50
            if (close[i] > upper_bb[i] and 
                squeeze_filter[i] and 
                vol_ok and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB AND squeeze AND 12h volume spike AND below 1d EMA50
            elif (close[i] < lower_bb[i] and 
                  squeeze_filter[i] and 
                  vol_ok and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-period SMA (mean reversion)
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-period SMA (mean reversion)
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals