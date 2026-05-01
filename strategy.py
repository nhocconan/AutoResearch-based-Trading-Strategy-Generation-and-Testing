#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX regime filter.
# Long when: price breaks above Donchian upper band AND 1d volume > 1.5x 20-period average AND ADX(14) > 25 (trending market).
# Short when: price breaks below Donchian lower band AND 1d volume > 1.5x 20-period average AND ADX(14) > 25.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-50 trades/year.
# Donchian channels provide clear breakout levels, volume confirms institutional participation,
# ADX filter ensures we only trade in trending conditions to avoid choppy markets.
# Works in bull markets (breakouts continuation) and bear markets (breakdown continuation) by following the trend.

name = "4h_Donchian20_1dVolConfirm_ADXRegime_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume SMA(20)
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 1d data
    # ADX requires +DM, -DM, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no prior close
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (similar to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        smoothed = np.full_like(values, np.nan)
        smoothed[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
            else:
                smoothed[i] = np.nan
        return smoothed
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_di_14 = wilders_smoothing(plus_dm, 14) / atr_14 * 100
    minus_di_14 = wilders_smoothing(minus_dm, 14) / atr_14 * 100
    dx = np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100
    adx_14 = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 4h
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Donchian(20) on 4h data
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and ADX
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_sma_20 = vol_sma_20_aligned[i]
        curr_adx = adx_14_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirm = curr_volume > (curr_vol_sma_20 * 1.5)
        
        # ADX regime filter: trending market (ADX > 25)
        trending_regime = curr_adx > 25
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND volume confirm AND trending regime
            if (curr_close > curr_upper and 
                volume_confirm and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND volume confirm AND trending regime
            elif (curr_close < curr_lower and 
                  volume_confirm and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower (reversal) OR ADX drops below 20 (trend weak)
            if (curr_close < curr_lower or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper (reversal) OR ADX drops below 20 (trend weak)
            if (curr_close > curr_upper or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals