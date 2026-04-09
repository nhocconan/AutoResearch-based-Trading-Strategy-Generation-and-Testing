#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 12h ADX trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets
# 12h ADX filters for trending vs ranging regimes (ADX < 20 = range, ADX > 25 = trend)
# Volume confirmation ensures breakouts have participation
# Works in bull/bear: ADX regime filter adapts, Williams %R captures reversals in range, breakouts in trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_12h_williamsr_adx_volume_v1"
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
    
    # Load 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
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
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_12h + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_12h + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    def ewm_smoothing(values, period):
        return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values
    adx_12h = ewm_smoothing(dx, 14)
    
    # Align 12h ADX to 4h timeframe (wait for 12h bar close)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 4h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 4h average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        # Regime filter: ADX < 20 = ranging (mean revert), ADX > 25 = trending (follow momentum)
        ranging_regime = adx_12h_aligned[i] < 20
        trending_regime = adx_12h_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) in ranging OR Williams %R < -80 in trending (failed momentum)
            if ranging_regime and williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            elif trending_regime and williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) in ranging OR Williams %R > -20 in trending (failed momentum)
            if ranging_regime and williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            elif trending_regime and williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if ranging_regime:
                # Mean reversion at extremes in ranging market
                if williams_r[i] < -80 and volume_confirmed:  # Oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif williams_r[i] > -20 and volume_confirmed:  # Overbought -> short
                    position = -1
                    signals[i] = -0.25
            elif trending_regime:
                # Follow momentum in trending market
                if williams_r[i] > -20 and volume_confirmed:  # Breaking above overbought -> long
                    position = 1
                    signals[i] = 0.25
                elif williams_r[i] < -80 and volume_confirmed:  # Breaking below oversold -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals