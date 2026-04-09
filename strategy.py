#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation
# Camarilla pivots identify key support/resistance levels: H4, H3, L3, L4
# In trending markets (6h ADX > 25): breakout continuation at H4/L4 levels with volume spike
# In ranging markets (6h ADX < 20): mean reversion at H3/L3 levels with volume confirmation
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: breakout momentum in trends, mean reversion in ranges

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Based on previous day's range: H4/L4 = extreme breakout, H3/L3 = mean reversion zones
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_ = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels (using previous day's data)
    h4 = pivot + range_ * 1.1 / 2
    h3 = pivot + range_ * 1.1 / 4
    l3 = pivot - range_ * 1.1 / 4
    l4 = pivot - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels for current day)
    h4_aligned = align_htf_to_ltf(prices, df_1d[:-1], h4)  # shift by 1 to use previous day
    h3_aligned = align_htf_to_ltf(prices, df_1d[:-1], h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d[:-1], l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d[:-1], l4)
    
    # Calculate 6h ADX for regime filter
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = wilders_smoothing(tr, 14)
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    vol_confirm = vol_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 6h ADX
        trending_regime = adx[i] > 25
        ranging_regime = adx[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price falls below H3 or volume dries up
                if close[i] < h3_aligned[i] or vol_confirm[i] == False:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns above L3
                if close[i] > l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price rises above L3 or volume dries up
                if close[i] > l3_aligned[i] or vol_confirm[i] == False:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns below H3
                if close[i] < h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Breakout continuation with volume confirmation
                if close[i] > h4_aligned[i] and vol_confirm[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < l4_aligned[i] and vol_confirm[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion at H3/L3 levels with volume confirmation
                if close[i] > h3_aligned[i] and vol_confirm[i]:
                    position = -1
                    signals[i] = -0.25
                elif close[i] < l3_aligned[i] and vol_confirm[i]:
                    position = 1
                    signals[i] = 0.25
    
    return signals