#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above R1 AND 1d volume > 1.5x 20-period volume SMA AND 1d ADX > 25 (trending)
# Short when price breaks below S1 AND same filters
# Camarilla levels from 1d provide intraday support/resistance, volume confirms conviction, ADX ensures trend
# Position size 0.25 to limit drawdown. Target: 75-200 total trades over 4 years

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for Camarilla, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = pivot + 1.1 * (high_1d - low_1d) / 12.0
    s1 = pivot - 1.1 * (high_1d - low_1d) / 12.0
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: ADX (14-period) for trend regime ===
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0]) * -1  # inverted for calculation
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14 * 100
    minus_di_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14 * 100
    
    dx = np.abs(plus_di_14 - minus_di_14) / (np.abs(plus_di_14) + np.abs(minus_di_14)) * 100
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (need 20 for volume SMA, 14*3 for ADX smoothing)
    warmup = 40
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_14_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # Price breaks above R1 AND volume confirmation AND trend filter
        if (close[i] > r1_aligned[i]) and vol_confirm and trend_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below S1 AND volume confirmation AND trend filter
        elif (close[i] < s1_aligned[i]) and vol_confirm and trend_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0