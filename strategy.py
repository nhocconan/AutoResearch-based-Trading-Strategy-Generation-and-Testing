#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + 1d Volume Spike + ADX Regime Filter.
Long when Williams %R < -80 (oversold) + 1d volume > 1.5x 20-period average + 1d ADX < 25 (low trend strength = mean reversion favorable).
Short when Williams %R > -20 (overbought) + same volume/ADX conditions.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses 1d for volume spike and ADX regime, 4h for Williams %R.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filters (volume spike, ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate 4h Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr_14[i]) or 
            np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime and volume conditions
        adx_val = adx_14_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        vol_val = volume_1d_aligned[i]
        wr_val = wr_14[i]
        
        # Volume spike: current volume > 1.5x 20-period average
        volume_spike = vol_val > 1.5 * vol_ma_val
        
        # Low trend strength regime: ADX < 25 (favorable for mean reversion)
        low_trend = adx_val < 25
        
        # Williams %R signals
        oversold = wr_val < -80
        overbought = wr_val > -20
        exit_long = wr_val > -50
        exit_short = wr_val < -50
        
        if position == 0:
            # Long: Oversold + volume spike + low trend (mean reversion setup)
            if oversold and volume_spike and low_trend:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + volume spike + low trend (mean reversion setup)
            elif overbought and volume_spike and low_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum shifting)
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum shifting)
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dVolumeSpike_ADX_Regime"
timeframe = "4h"
leverage = 1.0