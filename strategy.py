#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# 1d volume spike confirms institutional participation in the move
# Chop regime filter adapts to market conditions: trending (follow Alligator) vs ranging (fade extremes)
# Designed for 12h timeframe to capture medium-term swings with low trade frequency
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30
# Works in bull/bear: Alligator catches trends, chop filter prevents whipsaws in ranges

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volume normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
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
    
    # Calculate 1d average volume (20-period) normalized by ATR
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    # Normalize volume by ATR to get volume volatility ratio
    vol_ratio_1d = np.where(atr_1d > 0, avg_volume_1d / atr_1d, np.nan)
    avg_vol_ratio_1d = pd.Series(vol_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar close)
    avg_vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) >= 8 else np.full_like(jaw, np.nan)
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) >= 5 else np.full_like(teeth, np.nan)
    lips_shifted = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) >= 3 else np.full_like(lips, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(avg_vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2.0 * 20-period average volume (from 1d data aligned)
        # Calculate 20-period average volume from 1d data
        volume_s_1d = pd.Series(df_1d['volume'].values)
        avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
        avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
        
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow Alligator), CHOP > 61.8 = range (fade extremes)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Teeth OR regime shifts to ranging
            if close[i] < teeth_shifted[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Teeth OR regime shifts to ranging
            if close[i] > teeth_shifted[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow Alligator alignment in trending regime
                # Bullish: Lips > Teeth > Jaw
                if lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Bearish: Lips < Teeth < Jaw
                elif lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at extremes in ranging regime
                # Long when price touches Jaw (support) and volume confirms
                if close[i] <= jaw_shifted[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                # Short when price touches Jaw (resistance) and volume confirms
                elif close[i] >= jaw_shifted[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals