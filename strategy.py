#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX regime filter.
- Primary timeframe: 1d for execution, HTF: 1w for volume confirmation and ADX trend strength.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (mean reversion at Donchian levels).
- Entry: Long when price breaks above upper Donchian(20) AND ADX > 25 (bullish breakout in trend).
         Short when price breaks below lower Donchian(20) AND ADX > 25 (bearish breakout in trend).
         In ranging (ADX < 20): Long when price touches lower Donchian AND reverses up (close > low).
                                Short when price touches upper Donchian AND reverses down (close < high).
- Exit: Opposite Donchian breakout or ADX regime shift to ranging.
- Volume confirmation: current 1w volume > 2.0 * 20-period 1w volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in both bull and bear markets via regime adaptation: breakouts in trends, mean reversion in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Donchian, volume, and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:  # Need enough for 20-period MA + 20-period Donchian + 14-period ADX
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1w
    # Upper band = highest high over past 20 weeks
    upper_band = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    # Lower band = lowest low over past 20 weeks
    lower_band = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA on 1w
    volume_ma = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    # Volume spike: current volume > 2.0 * 20-period volume MA
    volume_spike = df_1w['volume'].values > (2.0 * volume_ma)
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need enough 1w bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above upper band
                    if curr_close > upper:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower band
                    elif curr_close < lower:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when price touches lower band and shows reversal (close > low)
                    if curr_low <= lower and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper band and shows reversal (close < high)
                    elif curr_high >= upper and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below lower band OR ADX drops to ranging
            if curr_close < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper band OR ADX drops to ranging
            if curr_close > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wVolumeSpike_1wADXRegime_v1"
timeframe = "1d"
leverage = 1.0