#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; entries taken on reversals
# from extreme levels with volume confirmation and higher timeframe trend alignment.
# Works in bull/bear by only taking long in bullish 12h trend, short in bearish.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period) on 12h data
    williams_r = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        highest_high = np.max(high_12h[i-13:i+1])
        lowest_low = np.min(low_12h[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_12h[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral if no range
    
    # Align Williams %R to 6h timeframe (wait for 12h bar close)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 12h EMA trend filter (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5 x 24-period average (4 days of 6h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Williams %R thresholds
    oversold = -80
    overbought = -20
    
    # Warmup: need 12h data (13 bars for Williams %R), EMA (50), volume MA (24)
    start_idx = max(13, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above oversold with volume and bullish trend
            if wr > oversold and williams_r_aligned[i-1] <= oversold and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: Williams %R crosses below overbought with volume and bearish trend
            elif wr < overbought and williams_r_aligned[i-1] >= overbought and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R reaches overbought or trend turns bearish
            if wr >= overbought or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R reaches oversold or trend turns bullish
            if wr <= oversold or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_14_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0