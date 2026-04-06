#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# Long when price closes above 20-period EMA during bullish 4h trend with volume > 1.5x average.
# Short when price closes below 20-period EMA during bearish 4h trend with volume confirmation.
# Uses 4h EMA trend to avoid counter-trend trades. Volume filter ensures conviction.
# Target: 80-160 total trades over 4 years (20-40/year) for optimal frequency.

name = "1h_ema20_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 20-period EMA for momentum
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h trend filter: EMA50 direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_prev = np.roll(ema50_4h, 1)
    ema50_4h_prev[0] = np.nan
    ema50_4h_slope = ema50_4h - ema50_4h_prev
    ema4h_trend_up = align_htf_to_ltf(prices, df_4h, ema50_4h_slope > 0)
    ema4h_trend_down = align_htf_to_ltf(prices, df_4h, ema50_4h_slope < 0)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 4h trend data not available
        if np.isnan(ema4h_trend_up[i]) or np.isnan(ema4h_trend_down[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below EMA20 or 4h trend turns bearish
            if (close[i] < ema20[i] or 
                ema4h_trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above EMA20 or 4h trend turns bullish
            if (close[i] > ema20[i] or 
                ema4h_trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and 4h trend filter
            if volume_filter:
                # Long: price closes above EMA20 during bullish 4h trend
                if (close[i] > ema20[i] and 
                    ema4h_trend_up[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: price closes below EMA20 during bearish 4h trend
                elif (close[i] < ema20[i] and 
                      ema4h_trend_down[i]):
                    signals[i] = -0.20
                    position = -1
    
    return signals