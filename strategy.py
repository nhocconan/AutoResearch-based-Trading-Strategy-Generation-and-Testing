#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; entries taken when
# price reverses from extreme levels with volume and higher timeframe trend.
# Works in bull/bear by filtering entry direction with 1d EMA trend.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R for each 1d bar (14-period)
    willr_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            willr_14[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            willr_14[i] = -50  # neutral when range is zero
    
    # Align Williams %R to 4h timeframe (wait for 1d close)
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr_14)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0 x 24-period average (4 days of 4h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (14 bars), EMA (34), volume MA (24)
    start_idx = max(14, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(willr_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Williams %R levels: oversold < -80, overbought > -20
        oversold = willr_aligned[i] < -80
        overbought = willr_aligned[i] > -20
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        if position == 0:
            # Long: exit oversold with volume and bullish trend
            if oversold and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: exit overbought with volume and bearish trend
            elif overbought and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: re-enter overbought or trend turns bearish
            if willr_aligned[i] > -50 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: re-enter oversold or trend turns bullish
            if willr_aligned[i] < -50 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_14_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0