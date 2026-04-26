#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2
Hypothesis: Tighten entry conditions from prior version to reduce trade count and improve Sharpe. Uses same Camarilla R1/S1 breakout logic but adds stricter volume confirmation (3.0x median) and requires ADX > 25 for strong trend regime. Discrete position sizing 0.25 to minimize fee churn. Designed for fewer, higher-quality trades in both bull and bear markets via 1d EMA trend filter and regime filter.
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
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    # First day: use same values (will be filtered by min_periods later)
    close_1d_prev[0] = close_1d[0]
    high_1d_prev[0] = high_1d[0]
    low_1d_prev[0] = low_1d[0]
    
    camarilla_range = high_1d_prev - low_1d_prev
    r1 = close_1d_prev + camarilla_range * 1.1 / 12
    s1 = close_1d_prev - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: volume > 3.0x 20-period median volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (3.0 * vol_median_20)
    
    # ADX regime filter: ADX > 25 = strong trend (favor breakouts)
    # ADX calculation: +DI, -DI, DX, then smoothed
    period_adx = 14
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=period_adx, min_periods=period_adx).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period_adx, min_periods=period_adx).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period_adx, min_periods=period_adx).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx) | (plus_di + minus_di == 0), 0, dx)
    adx = pd.Series(dx).rolling(window=period_adx, min_periods=period_adx).mean().values
    adx_filter = adx > 25
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume median, 14 for ADX
    start_idx = max(34, 20, period_adx)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        adx_ok = adx_filter[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R1 with volume spike, uptrend (close > EMA34_1d), and strong trend regime
            long_entry = (close_val > r1_aligned[i]) and vol_spike and (close_val > ema_34_val) and adx_ok
            # Short: price breaks below S1 with volume spike, downtrend (close < EMA34_1d), and strong trend regime
            short_entry = (close_val < s1_aligned[i]) and vol_spike and (close_val < ema_34_val) and adx_ok
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price re-enters Camarilla (below S1)
            if close_val < ema_34_val or close_val < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or price re-enters Camarilla (above R1)
            if close_val > ema_34_val or close_val > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0