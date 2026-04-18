#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Choppiness Index
    def calculate_chop(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        hh = np.maximum.accumulate(high)
        ll = np.minimum.accumulate(low)
        sum_atr = np.full_like(tr, np.nan)
        range_hl = np.full_like(tr, np.nan)
        
        if len(tr) >= period:
            sum_atr[period] = np.nansum(atr[1:period+1])
            range_hl[period] = hh[period] - ll[period]
            
            for i in range(period+1, len(tr)):
                sum_atr[i] = sum_atr[i-1] + atr[i]
                hh[i] = max(hh[i-1], high[i])
                ll[i] = min(ll[i-1], low[i])
                range_hl[i] = hh[i] - ll[i]
        
        chop = np.full_like(tr, np.nan)
        valid = (sum_atr != 0) & (range_hl != 0) & ~np.isnan(sum_atr) & ~np.isnan(range_hl)
        chop[valid] = 100 * np.log10(sum_atr[valid] / range_hl[valid]) / np.log10(period)
        
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Calculate Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        if len(high) >= period:
            for i in range(period-1, len(high)):
                upper[i] = np.max(high[i - period + 1:i + 1])
                lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    upper_20, lower_20 = calculate_donchian(high_1d, low_1d, 20)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    upper_12h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower_20)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_12h[i]) or np.isnan(upper_12h[i]) or 
            np.isnan(lower_12h[i]) or np.isnan(ema_1w_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Choppiness regime: Chop > 61.8 = ranging (mean revert)
        ranging = chop_12h[i] > 61.8
        
        if position == 0:
            # Long: price touches lower Donchian band in ranging market
            if low[i] <= lower_12h[i] and vol_confirm and ranging:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper Donchian band in ranging market
            elif high[i] >= upper_12h[i] and vol_confirm and ranging:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches midpoint or Chop drops below 40 (trend emerging)
            midpoint = (upper_12h[i] + lower_12h[i]) / 2
            if high[i] >= midpoint or chop_12h[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches midpoint or Chop drops below 40 (trend emerging)
            midpoint = (upper_12h[i] + lower_12h[i]) / 2
            if low[i] <= midpoint or chop_12h[i] < 40:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Choppiness_Donchian_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0