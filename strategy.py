#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume spike and choppiness regime filter
# Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# In trending regimes (CHOP < 38.2): Alligator lines aligned (Jaw>Teeth>Lips for long, reverse for short)
# In ranging regimes (CHOP > 61.8): trade mean reversion at extreme deviations from Alligator
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: trend following catches sustained moves, chop filter avoids whipsaws

name = "4h_1d_alligator_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
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
    
    # Calculate 1d average volume (20-period) normalized by ATR
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(atr_1d > 0, avg_volume_1d / atr_1d, np.nan)
    avg_vol_ratio_1d = pd.Series(vol_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Calculate Williams Alligator on 1d median price
    median_price_1d = (high_1d + low_1d) / 2.0
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw_1d = pd.Series(jaw_1d).rolling(window=8, min_periods=8).mean().values  # Smoothed with 8-period
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_1d = pd.Series(teeth_1d).rolling(window=5, min_periods=5).mean().values  # Smoothed with 5-period
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_1d = pd.Series(lips_1d).rolling(window=3, min_periods=3).mean().values  # Smoothed with 3-period
    
    # Calculate 1d ATR-based deviation bands for ranging regime
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    upper_dev_1d = close_1d + 1.5 * atr_ma_1d
    lower_dev_1d = close_1d - 1.5 * atr_ma_1d
    
    # Align 1d indicators to 4h timeframe
    avg_vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    upper_dev_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_dev_1d)
    lower_dev_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_dev_1d)
    
    # Pre-compute volume confirmation array
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_confirmed = volume > 2.0 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(avg_vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or
            np.isnan(lips_1d_aligned[i]) or np.isnan(upper_dev_1d_aligned[i]) or
            np.isnan(lower_dev_1d_aligned[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Alligator lines diverge wrong way or we enter ranging regime
                if not (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i]) or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price rises above upper deviation or drops below lips
                if close[i] > upper_dev_1d_aligned[i] or close[i] < lips_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Alligator lines diverge wrong way or we enter ranging regime
                if not (jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i]) or ranging_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price drops below lower deviation or rises above lips
                if close[i] < lower_dev_1d_aligned[i] or close[i] > lips_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Enter long when Alligator aligned bullish with volume confirmation
                if jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short when Alligator aligned bearish with volume confirmation
                elif jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean reversion: buy near lower deviation, sell near upper deviation
                if close[i] <= lower_dev_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= upper_dev_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals