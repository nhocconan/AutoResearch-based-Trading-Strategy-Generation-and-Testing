#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + chop regime filter
# Uses 1d Camarilla levels (L3, H3) for mean reversion in range, breakout in trend
# Volume confirms momentum, chop filter avoids whipsaw in strong trends
# Targets 12-30 trades/year (50-120 over 4 years) to minimize fee drag
# Works in bull/bear by adapting to regime: mean revert in chop, breakout in trend

name = "12h_camarilla1d_vol_chop_v1"
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
    
    # 14-period ATR for stops and chop calculation
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for Camarilla levels (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i >= 1:  # need previous day's data
            # Previous day's OHLC
            phigh = high_1d[i-1]
            plow = low_1d[i-1]
            pclose = close_1d[i-1]
            range_ = phigh - plow
            
            if range_ > 0:
                camarilla_h3[i] = pclose + range_ * 1.1 / 6
                camarilla_l3[i] = pclose - range_ * 1.1 / 6
                camarilla_h4[i] = pclose + range_ * 1.1 / 2
                camarilla_l4[i] = pclose - range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Choppy market indicator (Ehlers Chop Index)
    chop = np.full(n, np.nan)
    if n >= 14:
        # True range sum over 14 periods
        tr_sum = np.full(n, np.nan)
        if len(tr) >= 14:
            tr_sum[13] = np.sum(tr[:14])
            for i in range(14, n):
                tr_sum[i] = tr_sum[i-1] + tr[i-1] - tr[i-14]
        
        # Highest high and lowest low over 14 periods
        max_high = np.full(n, np.nan)
        min_low = np.full(n, np.nan)
        for i in range(13, n):
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
        
        # Chop calculation: log10(tr_sum / (max_high - min_low)) * 100 / log10(14)
        for i in range(13, n):
            if max_high[i] > min_low[i]:
                chop[i] = np.log10(tr_sum[i] / (max_high[i] - min_low[i])) * 100 / np.log10(14)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Chop regime: > 61.8 = range (mean revert), < 38.2 = trending (breakout)
        chop_value = chop[i]
        is_range = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit conditions depend on regime
            if is_range:
                # In range: exit at L3 or stoploss
                if close[i] <= l3_aligned[i] or close[i] < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trend: exit at H4 or stoploss
                if close[i] >= h4_aligned[i] or close[i] < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:  # short position
            # Exit conditions depend on regime
            if is_range:
                # In range: exit at H3 or stoploss
                if close[i] >= h3_aligned[i] or close[i] > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trend: exit at L4 or stoploss
                if close[i] <= l4_aligned[i] or close[i] > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Look for entries based on regime
            if is_range:
                # Mean reversion in range: buy at L3, sell at H3
                if close[i] <= l3_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] >= h3_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            elif is_trending:
                # Breakout in trend: buy above H4, sell below L4
                if close[i] > h4_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < l4_aligned[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: no trades
                signals[i] = 0.0
    
    return signals