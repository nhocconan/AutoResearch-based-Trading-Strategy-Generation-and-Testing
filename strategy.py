#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_Regime_v1
Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1-week EMA34 trend filter, volume spike confirmation, and choppiness regime filter.
Only takes breakouts in direction of 1-week trend when market is trending (CHOP < 38.2) to avoid false breakouts in ranging markets.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12-37 trades/year (50-150 over 4 years).
Works in bull/bear via 1w trend filter: only takes long breakouts in uptrend, short in downtrend.
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
    
    # Load 1w data ONCE before loop for HTF trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for HTF trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    htf_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R3_1d = typical_price_1d + (1.1/4) * (df_1d['high'] - df_1d['low'])
    S3_1d = typical_price_1d - (1.1/4) * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d.values)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d.values)
    
    # Calculate choppiness index on 1d data for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # We'll use a simplified version: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Rolling max(high) and min(low) over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_min_diff_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / max_min_diff_14) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((max_min_diff_14 > 0) & (~np.isnan(sum_atr_14)), chop, 50.0)
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime filter: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid breakouts)
    trending_regime = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA, 14*2 for chop)
    start_idx = max(34, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Only trade in trending regime
        if not trending_regime[i]:
            # In ranging market, exit any position and stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1w
            # Long breakout above R3 with volume spike
            if close[i] > R3_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S3 (reversal signal)
            elif position == 1 and close[i] < S3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1w
            # Short breakdown below S3 with volume spike
            if close[i] < S3_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R3 (reversal signal)
            elif position == -1 and close[i] > R3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_Regime_v1"
timeframe = "12h"
leverage = 1.0