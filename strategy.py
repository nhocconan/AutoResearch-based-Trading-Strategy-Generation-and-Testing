#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w volume confirmation + chop regime filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength
# 1w volume spike confirms institutional participation (avoids false signals)
# Choppiness index regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (follow Alligator)
# Works in bull/bear: regime filter adapts, Alligator catches strong moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1w_alligator_volume_chop_v1"
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
    
    # Load 1w data ONCE before loop for volume and chop calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w average volume (20-period)
    volume_1w = df_1w['volume'].values
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w Choppiness Index (CHOP)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
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
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1w - ll_1w
    chop_1w = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1w indicators to 12h timeframe (wait for 1w bar close)
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smoothed_moving_average(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        # First value is simple average
        sma = np.nanmean(values[:period])
        result = np.full(len(values), np.nan)
        result[period-1] = sma
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smoothed_moving_average(close, 13)
    teeth = smoothed_moving_average(close, 8)
    lips = smoothed_moving_average(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(avg_volume_1w_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1w average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1w_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow Alligator), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_1w_aligned[i] < 38.2
        ranging_regime = chop_1w_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Alligator lines converge (Teeth crosses Jaw) OR regime shifts to ranging
            if teeth[i] >= jaw[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (Teeth crosses Jaw) OR regime shifts to ranging
            if teeth[i] <= jaw[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow Alligator alignment in trending regime
                # Bullish: Lips > Teeth > Jaw (green alignment)
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Bearish: Lips < Teeth < Jaw (red alignment)
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at Alligator extremes in ranging regime
                # Buy when price touches Lows (Lips) and turns up
                if close[i] < lips[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Sell when price touches Highs (Lips) and turns down
                elif close[i] > lips[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals