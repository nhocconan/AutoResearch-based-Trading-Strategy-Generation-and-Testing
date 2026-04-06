#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with 1d trend filter and volume confirmation.
# Fisher Transform identifies turning points in price with clear -1.5/+1.5 thresholds.
# Uses 1d EMA50 as trend filter: long only in uptrend, short only in downtrend.
# Volume filter ensures sufficient participation for reliable signals.
# Works in both bull (trend-following) and bear (counter-trend at extremes) markets.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled risk.

name = "6h_fisher1d_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align EMA50 to 6bars (shifted by 1 day for prior day's trend)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Ehlers Fisher Transform on 6h prices
    # Price normalization: (Price - MinL) / (MaxH - MinL) * 2 - 1
    # Use 10-period lookback as recommended by Ehlers
    lookback = 10
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Normalized price in [-1, 1]
    normalized_price = (close - lowest_low) / range_hl * 2 - 1
    normalized_price = np.clip(normalized_price, -0.999, 0.999)  # Prevent log domain issues
    
    # Fisher Transform formula
    # Fisher = 0.5 * ln((1 + x) / (1 - x)) where x is normalized price
    # Smooth with 3-period EMA as per Ehlers
    fisher_raw = 0.5 * np.log((1 + normalized_price) / (1 - normalized_price))
    fisher = pd.Series(fisher_raw).ewm(span=3, adjust=False).mean().values
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(ema50_aligned[i]) or np.isnan(fisher[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: Fisher crosses below -1.5 (reversal signal) or trend breaks
            if fisher[i] < -1.5 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Fisher crosses above +1.5 (reversal signal) or trend breaks
            if fisher[i] > 1.5 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if vol_filter:
                # Long: Fisher crosses above -1.5 in uptrend (bullish reversal)
                if fisher[i] > -1.5 and fisher[i-1] <= -1.5 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Fisher crosses below +1.5 in downtrend (bearish reversal)
                elif fisher[i] < 1.5 and fisher[i-1] >= 1.5 and downtrend:
                    signals[i] = -0.25
                    position = -1
    
    return signals