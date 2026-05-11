#!/usr/bin/env python3
# 6h_1w_1d_VolumeBreakout_RSI_Filter
# Hypothesis: Breakout above 1d high or below 1d low on 6h timeframe with weekly trend filter, volume confirmation, and RSI filter to avoid overextended moves. Designed to work in bull (breakouts in uptrend) and bear (breakdowns in downtrend) with low trade frequency.

name = "6h_1w_1d_VolumeBreakout_RSI_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: 50 EMA slope
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_50_1w = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    ema_slope_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_50_1w)
    
    # Daily high and low for breakout levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_high_prev = np.roll(daily_high, 1)
    daily_low_prev = np.roll(daily_low, 1)
    daily_high_prev[0] = daily_high[0]
    daily_low_prev[0] = daily_low[0]
    
    # Align daily levels to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high_prev)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low_prev)
    
    # RSI(14) on 6s for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / loss_ma
    rs[loss_ma == 0] = 0
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i]) or
            np.isnan(ema_slope_50_1w_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from weekly EMA50 slope
        bullish_trend = ema_slope_50_1w_aligned[i] > 0
        bearish_trend = ema_slope_50_1w_aligned[i] < 0
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above previous day's high in bullish trend with volume surge and not overbought
            if (close[i] > daily_high_aligned[i] and 
                bullish_trend and 
                volume_surge and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low in bearish trend with volume surge and not oversold
            elif (close[i] < daily_low_aligned[i] and 
                  bearish_trend and 
                  volume_surge and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: price closes below previous day's low or RSI overbought
                if close[i] < daily_low_aligned[i] or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price closes above previous day's high or RSI oversold
                if close[i] > daily_high_aligned[i] or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals