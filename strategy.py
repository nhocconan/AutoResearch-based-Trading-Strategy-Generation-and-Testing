#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ATR volatility filter and 1w volume regime filter
# Long when price breaks above 12h Camarilla R3 level AND 1d ATR(14) > 1.2x 50-period median AND 1w volume > 1.3x 20-period average
# Short when price breaks below 12h Camarilla S3 level AND 1d ATR(14) > 1.2x 50-period median AND 1w volume > 1.3x 20-period average
# Exit when price crosses 12h Camarilla pivot point (mean reversion) OR 1d ATR(14) < 0.8x 50-period median (low volatility)
# Uses 12h primary timeframe with 1d for ATR volatility filter (adapts to changing market conditions) and 1w for volume regime (institutional participation)
# Higher timeframe volume filter ensures breakouts have sustained conviction, reducing false signals in low-volume environments
# ATR-based volatility filter prevents entries during excessively choppy or stagnant periods
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dATR_VolRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    
    # Calculate 50-period median of ATR for adaptive threshold
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).median().values
    atr_threshold = 1.2 * atr_ma_50  # Volatility expansion threshold
    low_vol_threshold = 0.8 * atr_ma_50  # Low volatility exit threshold
    
    # Align ATR and thresholds to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    low_vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, low_vol_threshold)
    
    # Calculate 1w volume regime filter
    volume_1w = df_1w['volume'].values
    vol_ma_20w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1w > (1.3 * vol_ma_20w)  # High volume regime
    
    # Align volume regime to 12h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime)
    
    # Calculate 1d Camarilla levels (based on previous 1d bar)
    camarilla_r3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_s3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3  # Standard pivot point
    
    # Align to 12h timeframe (using previous 1d bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_threshold_aligned[i]) or 
            np.isnan(low_vol_threshold_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility and regime filters
        vol_filter = atr_1d_aligned[i] > atr_threshold_aligned[i]
        regime_filter = vol_regime_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volatility expansion AND high volume regime
            if (close[i] > camarilla_r3_aligned[i] and 
                vol_filter and 
                regime_filter):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volatility expansion AND high volume regime
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_filter and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion) OR low volatility regime
            if close[i] < camarilla_pivot_aligned[i] or atr_1d_aligned[i] < low_vol_threshold_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion) OR low volatility regime
            if close[i] > camarilla_pivot_aligned[i] or atr_1d_aligned[i] < low_vol_threshold_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals