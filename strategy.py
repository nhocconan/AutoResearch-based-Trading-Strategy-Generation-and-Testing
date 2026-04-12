#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: Trade breakouts from daily Camarilla pivot levels with volume confirmation and 12h ADX regime filter.
Long when price breaks above daily H4 with volume > 1.5x average and 12h ADX > 25 (trending).
Short when price breaks below daily L4 with volume > 1.5x average and 12h ADX > 25.
Exit when price returns to daily pivot (H5/L5) or ADX drops below 20.
Uses discrete position sizing (0.25) to minimize churn. Designed for 15-25 trades/year.
Works in bull (breakouts continue) and bear (breakouts fail, reverse at pivots) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h5 = np.full_like(close_1d, np.nan)
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    camarilla_l5 = np.full_like(close_1d, np.nan)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        # Camarilla formulas using previous day's OHLC
        if i > 0:
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
                camarilla_pivot[i] = (ph + pl + pc) / 3
                range_val = ph - pl
                camarilla_h4[i] = pc + range_val * 1.1 / 2
                camarilla_l4[i] = pc - range_val * 1.1 / 2
                camarilla_h5[i] = pc + range_val * 1.1
                camarilla_l5[i] = pc - range_val * 1.1
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # === 12H INDICATORS: ADX FOR REGIME FILTER ===
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(low)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above daily H4 with volume and trend
        long_signal = (close[i] > h4_aligned[i] and 
                      strong_volume and 
                      trending)
        
        # Short: price breaks below daily L4 with volume and trend
        short_signal = (close[i] < l4_aligned[i] and 
                       strong_volume and 
                       trending)
        
        # Exit: price returns to H5/L5 or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < h5_aligned[i] or adx[i] < 20))
        exit_short = (position == -1 and 
                     (close[i] > l5_aligned[i] or adx[i] < 20))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals