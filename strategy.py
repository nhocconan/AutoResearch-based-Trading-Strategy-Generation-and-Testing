#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Mean Reversion with 4h Trend Filter and Volume Spike
# Hypothesis: In mean-reverting markets (range-bound), price reverts to the 1h VWAP
# when the 4h trend is neutral or opposing. Volume spikes confirm mean reversion
# opportunities. Works in both bull and bear markets by avoiding strong trends.
# Targets 15-35 trades/year with strict entry conditions to avoid overtrading.

name = "1h_mean_reversion_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, close)
    
    # 4h ADX for trend strength (avoid strong trends)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Calculate Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) / period
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align 4h ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume spike detector: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    # Session filter: 08-20 UTC (inclusive)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for ADX and volume MA
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Avoid strong trends: only trade when ADX < 25 (no strong trend)
        if adx_aligned[i] >= 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP as percentage
        vwap_dev = (close[i] - vwap[i]) / vwap[i]
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP OR volume spike fades
            if vwap_dev <= 0.005:  # Within 0.5% of VWAP
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price returns to VWAP OR volume spike fades
            if vwap_dev >= -0.005:  # Within 0.5% of VWAP
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: price below VWAP + volume spike + mean reversion setup
            if (vwap_dev < -0.015 and  # More than 1.5% below VWAP
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short: price above VWAP + volume spike + mean reversion setup
            elif (vwap_dev > 0.015 and  # More than 1.5% above VWAP
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals