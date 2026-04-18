#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter_v1
Hypothesis: 4-hour Donchian(20) breakouts with volume confirmation (>1.5x 20-period avg) and ADX(14) > 25 for trend strength. 
Long when price breaks above upper band, short when breaks below lower band. 
Exit when price returns to midpoint or ADX drops below 20. 
Uses daily timeframe for context but primary signals on 4h. 
Designed for low trade frequency (target 20-40/year) to minimize fee impact and work in both bull/bear markets via trend filter.
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
    
    # Get 4h data for primary calculations
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_band = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_band = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle_band = (upper_band + lower_band) / 2
    
    # 4h ADX for trend strength
    # True Range
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(np.roll(close_4h, 1) - low_4h)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]
    
    # Directional Movement
    up_move = np.maximum(high_4h - np.roll(high_4h, 1), 0)
    down_move = np.maximum(np.roll(low_4h, 1) - low_4h, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    tr_smooth[tr_period] = np.nansum(tr[1:tr_period+1]) if not np.isnan(tr).all() else 0
    for i in range(tr_period + 1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
    
    up_smooth = np.zeros_like(up_move)
    down_smooth = np.zeros_like(down_move)
    up_smooth[tr_period] = np.nansum(up_move[1:tr_period+1]) if not np.isnan(up_move).all() else 0
    down_smooth[tr_period] = np.nansum(down_move[1:tr_period+1]) if not np.isnan(down_move).all() else 0
    for i in range(tr_period + 1, len(up_move)):
        up_smooth[i] = up_smooth[i-1] - (up_smooth[i-1] / tr_period) + up_move[i]
        down_smooth[i] = down_smooth[i-1] - (down_smooth[i-1] / tr_period) + down_move[i]
    
    # Directional Indicators
    plus_di = 100 * up_smooth / tr_smooth
    minus_di = 100 * down_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX
    adx_period = 14
    adx = np.zeros_like(dx)
    adx[2*adx_period] = np.nanmean(dx[adx_period:2*adx_period+1]) if not np.isnan(dx).all() else 0
    for i in range(2*adx_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h data to 4h timeframe (no alignment needed as we're using 4h data directly)
    # But we need to align to the lower timeframe if we were using lower TF data
    # Since we're generating signals at 4h frequency, we use the 4h arrays directly
    
    # However, prices dataframe is at 1min? No - it's at the strategy's timeframe
    # The prices dataframe passed in is at the timeframe specified (4h in this case)
    # So we can use the values directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2*adx_period)  # need enough for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 for strong trend
        trend_filter = adx[i] > 25
        
        if position == 0:
            # Long: price breaks above upper band with volume and trend
            if close[i] > upper_band[i] and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume and trend
            elif close[i] < lower_band[i] and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or trend weakens
            if close[i] < middle_band[i] or adx[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or trend weakens
            if close[i] > middle_band[i] or adx[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0