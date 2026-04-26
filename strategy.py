#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_20_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Combine ADX regime filter with Donchian(20) breakout and 1-week trend on 6h timeframe.
- Use ADX(14) to detect trending (ADX>25) vs ranging (ADX<20) markets
- In trending markets (ADX>25): trade Donchian(20) breakouts in direction of 1-week trend
- In ranging markets (ADX<20): fade Donchian(20) breaks at R4/S4 Camarilla levels from 1d
- Volume confirmation (>1.5x 20-period average) required for all entries
- Designed for low trade frequency (12-37/year) by requiring multiple confluence factors
- Works in bull/bear via 1-week trend filter for breakouts and mean reversion in ranges
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    htf_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # ADX calculation (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)
    
    period = 14
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_di[period] = 100 * np.mean(plus_dm[1:period+1]) / atr[period]
    minus_di[period] = 100 * np.mean(minus_dm[1:period+1]) / atr[period]
    
    # Wilder's smoothing
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period
        minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period*2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX smoothing
    adx[period*2] = np.mean(dx[period+1:period*2+1])
    for i in range(period*2+1, n):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Camarilla R4/S4 from 1d for ranging market reversals
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R4_1d = typical_price_1d + (1.1/2) * (df_1d['high'] - df_1d['low'])
    S4_1d = typical_price_1d - (1.1/2) * (df_1d['high'] - df_1d['low'])
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d.values)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 2*period for ADX, 20 for Donchian/volume)
    start_idx = max(2*period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(R4_1d_aligned[i]) or 
            np.isnan(S4_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime detection
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        if is_trending:
            # Trending market: Donchian breakout with 1-week trend filter
            if htf_trend[i] == 1:  # Uptrend on 1w
                # Long breakout above Donchian high with volume spike
                if close[i] > donchian_high[i] and volume_spike:
                    if position != 1:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.25
                # Exit long if price falls below Donchian low
                elif position == 1 and close[i] < donchian_low[i]:
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
                # Short breakdown below Donchian low with volume spike
                if close[i] < donchian_low[i] and volume_spike:
                    if position != -1:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = -0.25
                # Exit short if price rises above Donchian high
                elif position == -1 and close[i] > donchian_high[i]:
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
        elif is_ranging:
            # Ranging market: fade Donchian breaks at Camarilla R4/S4
            # Long when price breaks above Donchian high but reverses below R4
            if position == 0 and close[i] > donchian_high[i] and volume_spike:
                # Potential false breakout - wait for reversal
                if close[i] < R4_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    # Hold flat
                    signals[i] = 0.0
            # Short when price breaks below Donchian low but reverses above S4
            elif position == 0 and close[i] < donchian_low[i] and volume_spike:
                # Potential false breakdown - wait for reversal
                if close[i] > S4_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    # Hold flat
                    signals[i] = 0.0
            # Exit long if price rises above R4 (failure of mean reversion)
            elif position == 1 and close[i] > R4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Exit short if price falls below S4 (failure of mean reversion)
            elif position == -1 and close[i] < S4_1d_aligned[i]:
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
            # Transition regime (ADX between 20-25): hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_Donchian_20_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0