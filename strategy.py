#!/usr/bin/env python3
"""
6h Williams %R + Volume + Trend Filter
Hypothesis: Williams %R identifies overbought/oversold conditions. Combined with volume confirmation
and 1-day EMA trend filter, it captures reversals in both bull and bear markets. Williams %R is
effective in ranging and trending markets, providing clear entry/exit signals.
Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williamsr_volume_trend_v1"
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
    
    # 14-period Williams %R
    williams_r = np.full(n, np.nan)
    if n >= 14:
        for i in range(14, n):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                williams_r[i] = -50  # neutral when range is zero
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 1-day EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(atr[i]) or 
            np.isnan(trend_bias_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R crosses above -20 (overbought) OR against trend
            # Stoploss: price drops 2*ATR below entry
            if (williams_r[i] > -20 or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Williams %R crosses below -80 (oversold) OR against trend
            # Stoploss: price rises 2*ATR above entry
            if (williams_r[i] < -80 or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Long: Williams %R crosses above -80 from below (oversold bounce) with volume and trend alignment
                williams_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80
                # Short: Williams %R crosses below -20 from above (overbought rejection) with volume and trend alignment
                williams_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20
                
                # Long: oversold bounce with volume and bullish trend
                if williams_cross_up and volume_filter and trend_bias_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: overbought rejection with volume and bearish trend
                elif williams_cross_down and volume_filter and trend_bias_aligned[i] == -1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals