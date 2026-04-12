#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter
    # Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
    # Regime: ADX(14) > 25 = trending (follow Elder Ray signals), ADX < 20 = ranging (fade extremes)
    # Volume > 1.3x 20-period MA confirms momentum
    # Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power and Bear Power
    bull_power = high_1d - ema_13_1d  # High - EMA13
    bear_power = ema_13_1d - low_1d   # EMA13 - Low
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX components: +DM, -DM, TR
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Wilder's smoothing: result[i] = (result[i-1] * (period-1) + data[i]) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 1d volume > 1.3x 20-period MA
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    vol_ratio_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        if vol_ma_20_1d[i] > 0:
            vol_ratio_1d[i] = volume_1d[i] / vol_ma_20_1d[i]
        else:
            vol_ratio_1d[i] = 1.0
    
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        # Volume confirmation
        vol_confirmed = vol_ratio_1d_aligned[i] > 1.3
        
        # Elder Ray signals
        strong_bull = bull_power_aligned[i] > 0 and bull_power_aligned[i] > bear_power_aligned[i]
        strong_bear = bear_power_aligned[i] > 0 and bear_power_aligned[i] > bull_power_aligned[i]
        
        # Entry logic: regime-dependent
        if trending and vol_confirmed:
            # Trending market: follow Elder Ray
            long_entry = strong_bull
            short_entry = strong_bear
        elif ranging and vol_confirmed:
            # Ranging market: fade extremes (contrarian)
            long_entry = bear_power_aligned[i] > 0  # Oversold: bear power positive but weakening
            short_entry = bull_power_aligned[i] > 0  # Overbought: bull power positive but weakening
        else:
            # Transition regime: no trades
            long_entry = False
            short_entry = False
        
        # Exit conditions: opposite signal or power divergence
        long_exit = strong_bear or (position == 1 and bear_power_aligned[i] > bull_power_aligned[i])
        short_exit = strong_bull or (position == -1 and bull_power_aligned[i] > bear_power_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_vol_v1"
timeframe = "6h"
leverage = 1.0