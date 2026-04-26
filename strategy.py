#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ADX_Regime
Hypothesis: Daily timeframe strategy using weekly Camarilla levels for structure, 1w EMA50 for trend filter, volume confirmation (>2.0x 20-bar MA), and ADX>25 regime filter. Targets 15-25 trades/year by requiring confluence of weekly trend, daily breakout, volume spike, and strong trending regime. Discrete sizing (±0.25) minimizes fee churn. Works in bull (breakouts with trend) and bear (mean reversion at extremes in ranging markets via regime filter).
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
    
    # Load weekly data ONCE before loop for trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for higher-timeframe trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate previous weekly bar's Camarilla levels (using weekly data)
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous weekly bar's high, low, close for Camarilla calculation
    prev_high = df_1w['high'].shift(1).values  # Shift to get previous weekly bar
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ADX regime filter: only trade in strongly trending markets (ADX > 25)
    # Calculate ADX on daily data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > high[i-1] - low[i] else 0
        minus_dm[i] = max(high[i-1] - low[i], 0) if high[i-1] - low[i] > high[i] - high[i-1] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_smooth = np.sum(plus_dm[1:period+1])
    minus_dm_smooth = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, n):
        atr[i] = atr[i-1] * (1 - alpha) + alpha * tr[i]
        plus_dm_smooth = plus_dm_smooth * (1 - alpha) + alpha * plus_dm[i]
        minus_dm_smooth = minus_dm_smooth * (1 - alpha) + alpha * minus_dm[i]
        plus_di[i] = 100 * plus_dm_smooth / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_smooth / atr[i] if atr[i] != 0 else 0
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
    
    # Smooth DX to get ADX
    adx[period*2] = np.mean(dx[period+1:period*2+1]) if len(dx[period+1:period*2+1]) > 0 else 0
    for i in range(period*2+1, n):
        adx[i] = adx[i-1] * (1 - alpha) + alpha * dx[i]
    
    adx_aligned = adx  # Already on daily timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size: 25% of capital
    
    # Warmup: max of calculations (20 for volume MA, 1 for shift, 50 for EMA, 28 for ADX)
    start_idx = max(20, 1, 50, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        
        # Determine weekly trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Regime filter: only trade in strongly trending markets (ADX > 25)
        trending_regime = adx_val > 25
        
        # Entry conditions: price breaks above/below Camarilla levels in direction of weekly trend with volume confirmation and trending regime
        long_entry = (close_val > r1_val) and bullish_1w and vol_spike and trending_regime
        short_entry = (close_val < s1_val) and bearish_1w and vol_spike and trending_regime
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal or regime change
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r1_val or not bullish_1w or not trending_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s1_val or not bearish_1w or not trending_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ADX_Regime"
timeframe = "1d"
leverage = 1.0