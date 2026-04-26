#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_VolumeRegime_v1
Hypothesis: On 4h timeframe, enter long when price touches Camarilla S3 level AND 1d trend is up (close > EMA34) AND volume > 1.8x 20-period average AND choppiness index < 40 (trending market). Enter short when price touches Camarilla R3 level AND 1d trend is down (close < EMA34) AND volume > 1.8x 20-period average AND choppiness index < 40. Uses discrete sizing (0.0, ±0.25) to limit fee churn. Camarilla levels from 1d provide strong support/resistance, volume spike confirms breakout validity, 1d trend filter ensures alignment with higher timeframe momentum, and chop filter avoids range-bound whipsaws. Designed to generate ~20-30 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # need at least previous day
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values  # raw 1d close for Camarilla calculation
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high - low)/4
    # S3 = close - 1.1*(high - low)/4
    # Using previous completed 1d bar to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    
    # First bar has no previous day, set to NaN
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range / 4
    s3 = prev_close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma
    
    # Choppiness index: CHOP < 40 = trending market (avoid ranging)
    def choppiness_index(high, low, close, window=14):
        """Calculate Choppiness Index"""
        atr = np.zeros_like(high)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if np.isnan(atr[i]) or atr[i] == 0 or np.isnan(max_high[i]) or np.isnan(min_low[i]):
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(sum(atr[max(0, i-window+1):i+1]) / (max_high[i] - min_low[i])) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    chop_filter = chop < 40  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup, volume MA warmup, and chop warmup
    start_idx = max(34, 20, 14)  # EMA34 needs 34, volume MA needs 20, chop needs 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Touch conditions (price near Camarilla levels)
        touch_s3 = low[i] <= s3_aligned[i] * 1.001  # allow small buffer
        touch_r3 = high[i] >= r3_aligned[i] * 0.999  # allow small buffer
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: touch S3 + volume spike + 1d uptrend + trending market
            long_signal = touch_s3 and volume_spike[i] and trend_uptrend and chop_filter[i]
            
            # Short: touch R3 + volume spike + 1d downtrend + trending market
            short_signal = touch_r3 and volume_spike[i] and trend_downtrend and chop_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price moves above midpoint OR trend change to downtrend OR chop becomes high (ranging)
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if close[i] > midpoint or not trend_uptrend or chop_filter[i] == False:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price moves below midpoint OR trend change to uptrend OR chop becomes high (ranging)
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if close[i] < midpoint or not trend_downtrend or chop_filter[i] == False:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0