#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with weekly trend filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13. 
# Weekly trend filter (EMA50 slope) ensures trades align with higher timeframe momentum.
# Volume confirmation filters for institutional participation.
# Designed for 6h timeframe to target 50-150 trades over 4 years with controlled frequency.
# Works in bull markets (buy bullish dips) and bear markets (sell bearish rallies).

name = "6h_elderray1w_vol_v1"
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
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray (13-period)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly EMA50 for trend filter (slope)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema50_1w = close_1w_s.ewm(span=50, adjust=False).mean().values
    # EMA50 slope: positive = uptrend, negative = downtrend
    ema50_slope = np.zeros_like(ema50_1w)
    ema50_slope[1:] = ema50_1w[1:] - ema50_1w[:-1]
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope)
    
    # Weekly volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.zeros_like(vol_1w)
    for i in range(19, len(vol_1w)):  # 20-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-19:i+1])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (EMA13 needs 13, EMA50 slope needs 51, vol needs 20)
    start = max(13, 51, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_slope_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Trend filter: weekly EMA50 slope
        uptrend = ema50_slope_aligned[i] > 0
        downtrend = ema50_slope_aligned[i] < 0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power turns positive (selling pressure) or stoploss
            if (bear_power[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns negative (buying pressure) or stoploss
            if (bull_power[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter
            if volume_filter:
                # Long: bullish power positive in uptrend OR extreme bullish power
                if (bull_power[i] > 0 and uptrend) or (bull_power[i] > np.percentile(bull_power[max(0,i-100):i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bearish power negative in downtrend OR extreme bearish power
                elif (bear_power[i] < 0 and downtrend) or (bear_power[i] < np.percentile(bear_power[max(0,i-100):i+1], 20)):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals