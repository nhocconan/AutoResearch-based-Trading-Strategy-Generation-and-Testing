#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation.
# Long when: Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average
# Short when: Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average
# Uses Elder Ray (Bull/Bear Power) to measure trend strength relative to EMA13, 1d ADX to filter for trending regimes only, volume spike for confirmation.
# Works in bull (catching strong uptrends) and bear (catching strong downtrends) by trading only when higher timeframe confirms trend strength.

name = "6h_ElderRay_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Elder Ray (EMA13 and high/low)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_6h = high - ema_13_6h
    bear_power_6h = ema_13_6h - low
    
    # Align Elder Ray to 6h primary timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_14 = 100 * dm_plus_14 / tr_14
    di_minus_14 = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus_14 - di_minus_14) / (di_plus_14 + di_minus_14)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h primary timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h volume average (20-period) for volume spike confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA13 and ADX (13+14+7 for smoothing)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_adx = adx_aligned[i]
        
        # Volume spike: current 6h volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # Elder Ray signals
        bullish_ray = (curr_bull_power > 0) and (curr_bear_power < 0)
        bearish_ray = (curr_bear_power > 0) and (curr_bull_power < 0)
        
        # 1d ADX trend filter: ADX > 25 indicates strong trend
        strong_trend = curr_adx > 25
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: bullish Elder Ray AND strong trend AND volume spike
            if (bullish_ray and 
                strong_trend and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: bearish Elder Ray AND strong trend AND volume spike
            elif (bearish_ray and 
                  strong_trend and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Elder Ray turns bearish OR ADX weakens (< 20) OR volume drops
            if (not bullish_ray or 
                curr_adx < 20 or 
                curr_vol < curr_vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Elder Ray turns bullish OR ADX weakens (< 20) OR volume drops
            if (not bearish_ray or 
                curr_adx < 20 or 
                curr_vol < curr_vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals