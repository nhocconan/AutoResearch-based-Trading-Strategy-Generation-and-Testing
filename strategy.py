#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + chop regime filter
# Williams Alligator identifies trend direction via smoothed medians (Jaw/Teeth/Lips)
# 1d volume spike confirms breakout authenticity and avoids false signals
# Choppiness index regime filter adapts to market conditions: trend follow when CHOP < 38.2, mean revert when CHOP > 61.8
# Works in bull/bear: regime filter prevents whipsaws in ranging markets, Alligator captures strong trends
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_alligator_volume_chop_v1"
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
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
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
    
    # Calculate 12h Williams Alligator (Smoothed Medians)
    median_12h = (high + low + close) / 3
    
    # Jaw: 13-period smoothed median, shifted by 8 bars
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # Shift forward by 8 bars
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period smoothed median, shifted by 5 bars
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # Shift forward by 5 bars
    teeth[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: 5-period smoothed median, shifted by 3 bars
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # Shift forward by 3 bars
    lips[:3] = np.nan  # First 3 values invalid after shift
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = strong trend (follow Alligator), CHOP > 61.8 = range (mean revert)
        strong_trend = chop_1d_aligned[i] < 38.2
        ranging_market = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Lips cross below Teeth OR regime shifts to ranging
            if lips[i] < teeth[i] or ranging_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Lips cross above Teeth OR regime shifts to ranging
            if lips[i] > teeth[i] or ranging_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if strong_trend:
                # Follow Alligator alignment in strong trend
                if lips[i] > teeth[i] > jaw[i] and volume_confirmed:  # Bullish alignment
                    position = 1
                    signals[i] = 0.25
                elif lips[i] < teeth[i] < jaw[i] and volume_confirmed:  # Bearish alignment
                    position = -1
                    signals[i] = -0.25
            elif ranging_market:
                # Mean revert at extremes in ranging market
                # Use price deviation from median as proxy for extremes
                if close[i] < np.min([jaw[i], teeth[i], lips[i]]) and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > np.max([jaw[i], teeth[i], lips[i]]) and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals