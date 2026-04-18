#!/usr/bin/env python3
"""
12h_LowVolatilityBreakout_v1
Strategy: 12h breakout from low volatility periods with trend filter and volume confirmation.
Long: Price breaks above Bollinger Upper Band after low volatility period in uptrend.
Short: Price breaks below Bollinger Lower Band after low volatility period in downtrend.
Designed for 12h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and volatility-based breakout logic.
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
    
    # Get daily data for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Bollinger Band Width for volatility regime
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Low volatility threshold (20th percentile of BB width)
    bb_width_20th = np.nanpercentile(bb_width[~np.isnan(bb_width)], 20) if np.sum(~np.isnan(bb_width)) > 0 else 0.02
    low_vol = bb_width_aligned < bb_width_20th
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(low_vol[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = high[i] > upper_bb_aligned[i] and close[i] > upper_bb_aligned[i]
        breakout_down = low[i] < lower_bb_aligned[i] and close[i] < lower_bb_aligned[i]
        
        if position == 0:
            # Long: uptrend + low volatility + volume + breakout up
            if uptrend and low_vol[i] and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + low volatility + volume + breakout down
            elif downtrend and low_vol[i] and vol_confirm and breakout_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volatility expansion, or breakout down
            if not uptrend or not low_vol[i] or breakout_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volatility expansion, or breakout up
            if not downtrend or not low_vol[i] or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_LowVolatilityBreakout_v1"
timeframe = "12h"
leverage = 1.0