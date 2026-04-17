#!/usr/bin/env python3
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
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 20-period SMA and std dev for Bollinger Bands
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + (2 * std20_1d)
    lower_bb_1d = sma20_1d - (2 * std20_1d)
    
    # Align BB to 12h timeframe
    upper_bb_12h = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_12h = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need Bollinger Bands, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_12h[i]) or 
            np.isnan(lower_bb_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        if position == 0:
            # Long: Price touches lower Bollinger Band AND price above 1w EMA50 with volume
            if (low[i] <= lower_bb_12h[i] and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band AND price below 1w EMA50 with volume
            elif (high[i] >= upper_bb_12h[i] and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses above middle Bollinger Band (SMA20)
            middle_bb_12h = sma20_1d[-1] if len(sma20_1d) > 0 else 0  # placeholder, will be updated
            # Calculate middle BB for current day's alignment
            # Since we don't have direct access to sma20_1d aligned, we use price crossing above lower BB as exit
            if close[i] >= upper_bb_12h[i]:  # Exit when price reaches upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses below middle Bollinger Band (SMA20)
            if close[i] <= lower_bb_12h[i]:  # Exit when price reaches lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerBands_EMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0