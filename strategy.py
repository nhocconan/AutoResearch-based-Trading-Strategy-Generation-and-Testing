#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with volume confirmation and weekly trend filter.
# Long when price breaks above upper Bollinger Band (20,2) with volume > 1.5x average and weekly close > EMA34.
# Short when price breaks below lower Bollinger Band with volume > 1.5x average and weekly close < EMA34.
# Exit when price returns to middle band or weekly trend reverses.
# Designed for ~10-20 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20,2) on daily close
    bb_period = 20
    bb_std = 2.0
    
    # Calculate rolling mean and std for Bollinger Bands
    bb_mean = np.full(n, np.nan)
    bb_std_dev = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        bb_mean[i] = np.mean(close[i-bb_period+1:i+1])
        bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    
    bb_upper = bb_mean + bb_std_dev * bb_std
    bb_lower = bb_mean - bb_std_dev * bb_std
    bb_middle = bb_mean
    
    # Get weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period BB and volume MA
    start_idx = max(bb_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from weekly EMA34
        bullish_trend = price > ema34_aligned[i]
        bearish_trend = price < ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB with volume and bullish trend
            if price > bb_upper[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower BB with volume and bearish trend
            elif price < bb_lower[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below middle band or trend turns bearish
            if price < bb_middle[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above middle band or trend turns bullish
            if price > bb_middle[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_BollingerBand_Breakout_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0