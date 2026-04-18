#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day trend filter and volume confirmation.
# Uses Williams %R(14) for overbought/oversold signals on 4h timeframe,
# daily EMA20 for trend direction, and volume spike for confirmation.
# Designed for mean reversion in ranging markets and trend exhaustion in trending markets.
# Target: 20-50 trades per year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(20) on daily data
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily EMA20 to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
    for i in range(14):
        highest_high[i] = np.max(high[:i+1])
        lowest_low[i] = np.min(low[:i+1])
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need daily EMA20, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above daily EMA20 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_20_1d_aligned[i]
        trend_down = close[i] < ema_20_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) with volume and in uptrend
            if (williams_r[i] < -80 and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) with volume and in downtrend
            elif (williams_r[i] > -20 and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or overbought
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or oversold
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_DailyEMA20_VolumeConfirm"
timeframe = "4h"
leverage = 1.0