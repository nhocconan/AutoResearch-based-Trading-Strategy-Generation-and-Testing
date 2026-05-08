#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour 3-bar breakout with 4-hour trend filter and volume confirmation
# Uses 4-hour close > SMA200 for trend direction, volume spike > 1.5x 20-period average
# Enters on break of highest high/lowest low of last 3 bars in trend direction
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Works in bull/bear via trend filter and volatility-based position sizing

name = "1h_3BarBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h SMA200 for trend filter
    close_4h = df_4h['close'].values
    sma_4h_200 = pd.Series(close_4h).rolling(window=200, min_periods=200).mean().values
    
    # Align 4h SMA to 1h timeframe
    sma_4h_200_aligned = align_htf_to_ltf(prices, df_4h, sma_4h_200)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for rolling calculations
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(sma_4h_200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction using 4h SMA200
        uptrend = close > sma_4h_200_aligned[i]
        downtrend = close < sma_4h_200_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for 3-bar breakout in trend direction
            if i >= 3:
                # For long: break above highest high of last 3 bars
                high_3bar = np.max(high[i-3:i])
                # For short: break below lowest low of last 3 bars
                low_3bar = np.min(low[i-3:i])
                
                if uptrend and high[i] > high_3bar and vol_filter:
                    signals[i] = 0.20
                    position = 1
                elif downtrend and low[i] < low_3bar and vol_filter:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price returns to lowest low of last 3 bars or trend reverses
            if i >= 3:
                low_3bar = np.min(low[i-3:i])
                if low[i] <= low_3bar or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to highest high of last 3 bars or trend reverses
            if i >= 3:
                high_3bar = np.max(high[i-3:i])
                if high[i] >= high_3bar or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals