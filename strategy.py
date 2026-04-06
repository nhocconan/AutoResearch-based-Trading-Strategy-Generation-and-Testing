#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with Donchian(20) breakout and volume confirmation.
# In low chop (< 38.2) = trending: breakout entries. In high chop (> 61.8) = ranging: mean-reversion at extremes.
# Uses 1-day ATR for volatility normalization and 1-week trend filter to avoid counter-trend trades.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in bull/bear via regime adaptation and volatility-adjusted position sizing.

name = "12h_chop_regime_breakout_v1"
timeframe = "12h"
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
    
    # 1-day ATR for volatility normalization (used in chop calculation)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR(14) for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1d[i] = np.mean(tr[i-13:i+1])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # 1-week close for trend filter (avoid counter-trend in strong trends)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_50_1w = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align 1d ATR and 1w SMA to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Choppiness Index (14-period) - measures trend vs range
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low))) / log10(n)
    # Simplified: using ATR sum over period vs range
    atr_14 = np.full(n, np.nan)
    for i in range(13, n):
        if i == 13:
            atr_14[i] = np.mean(tr[i-13:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Chop: 100 * log10(sum(ATR14 over 14) / (highest_high - lowest_low)) / log10(14)
    chop = np.full(n, 50.0)  # default neutral
    for i in range(27, n):  # need 14+14 for lookback
        sum_atr = np.sum(atr_14[i-13:i+1])
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        range_hl = highest_high - lowest_low
        if range_hl > 0 and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(range_hl)
        else:
            chop[i] = 50.0
    
    # 12-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 12h volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(28, n):  # start after chop warmup
        # Skip if required data not available
        if (np.isnan(chop[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filters
        chop_value = chop[i]
        is_trending = chop_value < 38.2  # strong trend
        is_ranging = chop_value > 61.8   # strong range
        
        # Trend filter: avoid counter-trend in strong weekly trends
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and volatility-based stoploss (2.0 * 1d ATR)
        atr_val = atr_1d_aligned[i]
        stop_distance = 2.0 * atr_val if not np.isnan(atr_val) else np.inf
        
        if position == 1:  # long position
            # Exit conditions
            exit_condition = False
            if is_ranging and close[i] < lowest_low[i] + 0.1 * (highest_high[i] - lowest_low[i]):
                exit_condition = True  # mean reversion at support in ranging
            elif close[i] < entry_price - stop_distance:
                exit_condition = True  # stoploss
            elif not weekly_uptrend and chop_value < 30:  # strong counter-trend signal
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions
            exit_condition = False
            if is_ranging and close[i] > highest_high[i] - 0.1 * (highest_high[i] - lowest_low[i]):
                exit_condition = True  # mean reversion at resistance in ranging
            elif close[i] > entry_price + stop_distance:
                exit_condition = True  # stoploss
            elif not weekly_downtrend and chop_value < 30:  # strong counter-trend signal
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime
            if volume_filter:
                if is_trending:
                    # Trending market: breakout entries
                    # Long: breakout above resistance with weekly uptrend filter
                    if (highest_high[i] > highest_high[i-1] and 
                        close[i] > highest_high[i] and weekly_uptrend):
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: breakdown below support with weekly downtrend filter
                    elif (lowest_low[i] < lowest_low[i-1] and 
                          close[i] < lowest_low[i] and weekly_downtrend):
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                elif is_ranging:
                    # Ranging market: mean reversion at extremes
                    # Long: near support with reversal signs
                    if (close[i] <= lowest_low[i] + 0.05 * (highest_high[i] - lowest_low[i]) and
                        close[i] > open[i] and  # bullish candle
                        lowest_low[i] == lowest_low[i-1]):  # tested support
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Short: near resistance with reversal signs
                    elif (close[i] >= highest_high[i] - 0.05 * (highest_high[i] - lowest_low[i]) and
                          close[i] < open[i] and  # bearish candle
                          highest_high[i] == highest_high[i-1]):  # tested resistance
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals