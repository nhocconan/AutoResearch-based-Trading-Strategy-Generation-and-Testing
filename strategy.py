#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal + 1d volume spike + chop regime filter
# Williams %R identifies overbought/oversold conditions for mean reversion
# 1d volume spike confirms institutional participation in reversals
# Chop regime filter adapts: CHOP > 61.8 = range (strong mean reversion), CHOP < 38.2 = trending (weaker signals)
# Works in bull/bear: mean reversion effective in ranging markets, volume confirms breakout authenticity
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - smoothed TR using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Williams %R (14-period)
    highest_high_12h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_12h = np.where((highest_high_12h - lowest_low_12h) != 0,
                              -100 * (highest_high_12h - close) / (highest_high_12h - lowest_low_12h),
                              -50)  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or
            np.isnan(williams_r_12h[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 = strong ranging (aggressive mean reversion)
        #              CHOP < 38.2 = trending (weaker mean reversion, needs stronger signal)
        strong_ranging = chop_1d_aligned[i] > 61.8
        weak_ranging = chop_1d_aligned[i] > 38.2 and chop_1d_aligned[i] <= 61.8
        trending = chop_1d_aligned[i] <= 38.2
        
        if position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) OR regime shifts to trending
            if williams_r_12h[i] > -20 or trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) OR regime shifts to trending
            if williams_r_12h[i] < -80 or trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime and Williams %R extremes
            if strong_ranging:
                # Strong ranging: aggressive mean reversion at extreme levels
                if williams_r_12h[i] <= -80 and volume_confirmed:  # Oversold
                    position = 1
                    signals[i] = 0.25
                elif williams_r_12h[i] >= -20 and volume_confirmed:  # Overbought
                    position = -1
                    signals[i] = -0.25
            elif weak_ranging:
                # Weak ranging: moderate mean reversion, needs volume confirmation
                if williams_r_12h[i] <= -85 and volume_confirmed:  # Deep oversold
                    position = 1
                    signals[i] = 0.25
                elif williams_r_12h[i] >= -15 and volume_confirmed:  # Deep overbought
                    position = -1
                    signals[i] = -0.25
            # In trending regime: no entries (avoid fading trends)
    
    return signals