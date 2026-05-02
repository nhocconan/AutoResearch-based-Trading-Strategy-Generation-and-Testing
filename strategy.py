#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13; ADX > 25 indicates trending market
# Volume spike (1.8x 20-period average) ensures strong participation
# Works in bull/bear markets by only taking trend-following entries when ADX confirms trend
# Targets 12-30 trades/year (50-120 total over 4 years) to minimize fee drag

name = "6h_ElderRay_12hADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend strength filter
    # ADX requires +DI, -DI, and TR calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_di_14 = wilders_smoothing(plus_dm, 14)
    minus_di_14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    dx = np.zeros_like(tr_14)
    mask = (plus_di_14 + minus_di_14) > 0
    dx[mask] = 100 * np.abs(plus_di_14[mask] - minus_di_14[mask]) / (plus_di_14[mask] + minus_di_14[mask])
    
    adx_14 = wilders_smoothing(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_14)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate volume spike (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA13, ADX and volume MA)
    start_idx = 60  # max(13 for EMA, 14*3 for ADX calc, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX indicates trending market (> 25)
        is_trending = adx_12h_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (bulls in control) + ADX trending + volume spike
            if bull_power[i] > 0 and is_trending and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + ADX trending + volume spike
            elif bear_power[i] < 0 and is_trending and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative OR ADX weakens (< 20) indicating trend end
            if bull_power[i] <= 0 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR ADX weakens (< 20) indicating trend end
            if bear_power[i] >= 0 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals