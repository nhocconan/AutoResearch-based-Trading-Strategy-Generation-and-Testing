#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Weekly Donchian breakouts capture major trend changes that work in both bull and bear markets.
Volume confirmation ensures institutional participation, while ADX filter avoids choppy markets.
Targets 15-25 trades/year to minimize fee drag. Uses 1d timeframe with 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly ADX for trend strength filter
    # Calculate +DM, -DM, TR
    up_move = np.diff(df_1w['high'].values, prepend=df_1w['high'].values[0])
    down_move = np.diff(df_1w['low'].values, prepend=df_1w['low'].values[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_period = 14
    atr_1w = wilder_smooth(tr, atr_period)
    plus_di_1w = 100 * wilder_smooth(plus_dm, atr_period) / atr_1w
    minus_di_1w = 100 * wilder_smooth(minus_dm, atr_period) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilder_smooth(dx_1w, atr_period)
    
    # Avoid division by zero
    adx_1w = np.where((plus_di_1w + minus_di_1w) == 0, 0, adx_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx = adx_1w_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and strong trend
            if price > donch_high[i] and vol_ok and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and strong trend
            elif price < donch_low[i] and vol_ok and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to Donchian low or trend weakens
            if price < donch_low[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to Donchian high or trend weakens
            if price > donch_high[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0