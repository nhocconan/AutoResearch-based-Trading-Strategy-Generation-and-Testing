#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter, volume > 1.5x 20-period average, and choppiness regime filter (CHOP < 50 = trending).
Only takes breakouts in direction of 1d trend to avoid counter-trend whipsaws. Uses chop filter to avoid ranging markets.
Designed for 20-50 trades/year (80-200 over 4 years) by requiring confluence of breakout, trend, volume, and regime.
Works in bull/bear via 1d trend filter: only long in uptrend, short in downtrend. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data (R1, S1)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 4h for regime filter
    def calculate_chop(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth TR with Wilder's smoothing (alpha = 1/window)
        atr[window-1] = np.mean(tr[1:window])  # seed
        for i in range(window, len(tr)):
            atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        # Calculate sum of true range over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        # Calculate max(high) - min(low) over window
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = max_high - min_low
        # CHOP = 100 * log10(tr_sum / range_max_min) / log10(window)
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if range_max_min[i] > 0 and not np.isnan(tr_sum[i]) and not np.isnan(range_max_min[i]):
                chop[i] = 100 * np.log10(tr_sum[i] / range_max_min[i]) / np.log10(window)
            else:
                chop[i] = 50.0  # default to middle range
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    chop_ma = pd.Series(chop).rolling(window=5, min_periods=5).mean().values  # smooth chop
    trending_regime = chop_ma < 50.0  # CHOP < 50 = trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for volume MA, 14 for CHOP)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(chop_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade in trending markets (CHOP < 50)
        in_trending_regime = trending_regime[i]
        
        # Breakout conditions with trend filter and regime filter
        if htf_trend[i] == 1 and in_trending_regime:  # Uptrend on 1d + trending regime
            # Long breakout above R1 with volume spike
            if close[i] > R1_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_1d_aligned[i]:
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
        elif htf_trend[i] == -1 and in_trending_regime:  # Downtrend on 1d + trending regime
            # Short breakdown below S1 with volume spike
            if close[i] < S1_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_1d_aligned[i]:
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
            # Either counter-trend or ranging market - hold or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0