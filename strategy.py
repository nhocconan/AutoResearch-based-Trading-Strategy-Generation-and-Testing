#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ADX_Regime_v2
Hypothesis: Refined version with stricter volume filter (>2.5x 20-bar MA) and ADX threshold (ADX>30) to reduce overtrading. Targets 15-30 trades/year for better test generalization. Uses discrete sizing (±0.25) to minimize fee churn. Long when price breaks above R3 in bullish 1d trend with volume confirmation and strong trending regime; short when breaks below S3 in bearish 1d trend.
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
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous 1d bar's Camarilla levels (using 1d data)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shift to get previous 1d bar
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.5x 20-period average (stricter filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # ADX regime filter: only trade in strongly trending markets (ADX > 30)
    # Calculate ADX on 4h data
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
    
    adx_aligned = adx  # Already on 4h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Reduced size to further limit drawdown
    
    # Warmup: max of calculations (20 for volume MA, 1 for shift, 34 for EMA, 28 for ADX)
    start_idx = max(20, 1, 34, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Regime filter: only trade in strongly trending markets (ADX > 30)
        trending_regime = adx_val > 30
        
        # Entry conditions: price breaks above/below Camarilla levels in direction of 1d trend with volume confirmation and trending regime
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike and trending_regime
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike and trending_regime
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal or regime change
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r3_val or not bullish_1d or not trending_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s3_val or not bearish_1d or not trending_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_ADX_Regime_v2"
timeframe = "4h"
leverage = 1.0