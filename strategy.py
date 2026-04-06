#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trading with 4h trend filter and 1d volume confirmation.
# Long when price breaks above 4h EMA(20) during bullish day (close > open) with volume > 1.5x 20-period average.
# Short when price breaks below 4h EMA(20) during bearish day (close < open) with volume confirmation.
# Uses 4h EMA for trend, 1d for daily bias, and volume for confirmation.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "1h_ema20_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA(20) for trend
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 4h EMA or daily trend data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below 4h EMA or daily turn bearish
            if (close[i] < ema_4h_aligned[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price crosses above 4h EMA or daily turn bullish
            if (close[i] > ema_4h_aligned[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: price crosses above 4h EMA during bullish day
                if (close[i] > ema_4h_aligned[i] and 
                    daily_bullish_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: price crosses below 4h EMA during bearish day
                elif (close[i] < ema_4h_aligned[i] and 
                      daily_bearish_aligned[i]):
                    signals[i] = -0.20
                    position = -1
    
    return signals