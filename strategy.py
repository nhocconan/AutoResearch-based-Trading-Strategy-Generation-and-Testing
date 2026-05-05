#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1d Volume Spike + Chop Regime Filter
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND Chop(14) > 61.8 (range)
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND Chop(14) > 61.8 (range)
# Exit when price retouches Camarilla Pivot Point (PP) OR Chop(14) < 38.2 (trend regime)
# Uses 12h primary timeframe with 1d HTF for volume and chop filter to avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Camarilla levels provide precise intraday support/resistance; volume spike confirms participation;
# chop filter ensures we only trade in ranging markets where mean reversion works best

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter"
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
    
    # Get 1d data ONCE before loop for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA for spike detection
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    if len(df_1d) >= 14:
        tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values).abs()
        tr2 = pd.Series(df_1d['high'].values - df_1d['close'].shift(1).values).abs()
        tr3 = pd.Series(df_1d['low'].values - df_1d['close'].shift(1).values).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        chop_denom = np.log10((max_high - min_low) / atr_14) * np.sqrt(14)
        chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
        chop_1d = 100 - (100 * np.log10(atr_14 * np.sqrt(14)) / chop_denom)
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, 50.0)  # neutral chop when insufficient data
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla levels are based on previous day's range
    # PP = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    # We need to shift these by 1 bar to avoid look-ahead (use previous day's levels)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # First bar has no previous day, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pp_1d = (prev_high + prev_low + prev_close) / 3.0
    r3_1d = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    s3_1d = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        in_range = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long conditions: price breaks above R3 AND volume spike AND in range
            if close[i] > r3_1d_aligned[i] and volume_spike_1d_aligned[i] and in_range:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND volume spike AND in range
            elif close[i] < s3_1d_aligned[i] and volume_spike_1d_aligned[i] and in_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches PP OR chop < 38.2 (trend regime)
            if close[i] <= pp_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches PP OR chop < 38.2 (trend regime)
            if close[i] >= pp_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals