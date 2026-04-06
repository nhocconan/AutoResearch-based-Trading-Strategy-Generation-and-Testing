#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R + Bollinger Bands with 1-day trend filter.
# Uses Williams %R(14) for overbought/oversold signals and Bollinger Bands(20,2) for volatility context.
# Trend filter from 1-day EMA(50) ensures trades align with higher timeframe direction.
# Designed for 6h timeframe to target 50-150 trades over 4 years with medium frequency.
# Works in bull/bear markets via mean reversion in ranging markets and trend following in trending markets.

name = "6h_williamsr_bbands_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA(50) on 1d close
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2/51) + (ema_50_1d[i-1] * 49/51)
    
    # Align 1-day EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Bollinger Bands(20,2) on 6h close
    sma_20 = np.full(n, np.nan)
    std_20 = np.full(n, np.nan)
    
    for i in range(19, n):
        sma_20[i] = np.mean(close[i-19:i+1])
        std_20[i] = np.std(close[i-19:i+1])
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: bullish if price > EMA50, bearish if price < EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R > -20 (overbought) or stoploss
            if (williams_r[i] > -20 or 
                close[i] < entry_price - 2.0 * (sma_20[i] - lower_band[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -80 (oversold) or stoploss
            if (williams_r[i] < -80 or 
                close[i] > entry_price + 2.0 * (upper_band[i] - sma_20[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on Williams %R extremes with trend filter
            if williams_r[i] < -80 and bullish_trend:  # oversold in uptrend -> long
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif williams_r[i] > -20 and bearish_trend:  # overbought in downtrend -> short
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals