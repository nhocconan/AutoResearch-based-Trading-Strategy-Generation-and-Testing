#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d volume confirmation + chop regime filter
# Uses 12h Donchian(20) for breakout entries, confirmed by 1d volume spike (>1.5x 20-period average)
# and 1d choppiness index (< 38.2 for trending markets). Exits on opposite Donchian break or
# chop regime shift (> 61.8). Designed for low trade frequency (12-25/year) to minimize fee drag.
# Works in bull/bear by only taking breakouts in trending regimes (chop < 38.2), avoiding whipsaws
# in ranging markets. Discrete sizing 0.25 balances return and drawdown.

name = "12h_Donchian20_1dVolConfirm_ChopRegime_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    # 1d choppiness index
    def choppiness_index(high, low, close, window=14):
        """Calculate Choppiness Index"""
        if len(high) < window:
            return np.full_like(high, np.nan)
        atr = np.zeros_like(high)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # First TR is just high-low
        tr[0] = tr1[0]
        # Sum of TR over window
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        # True range calculation for first element
        if window > 0:
            atr_sum[window-1:] = pd.Series(tr).rolling(window=window, min_periods=window).sum().values[window-1:]
        # Highest high and lowest low over window
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        # Avoid division by zero
        hh_ll = highest_high - lowest_low
        chop = np.where(
            (hh_ll != 0) & (atr_sum != 0),
            100 * np.log10(atr_sum / hh_ll) / np.log10(window),
            50.0  # neutral when undefined
        )
        return chop
    
    chop = choppiness_index(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align 1d indicators to 12h
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h Donchian channels (20-period)
    def donchian_channels(high, low, window=20):
        """Calculate Donchian channels"""
        if len(high) < window:
            return np.full_like(high, np.nan), np.full_like(low, np.nan)
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        # Fix first values
        upper[window-1:] = pd.Series(high).rolling(window=window, min_periods=window).max().values[window-1:]
        lower[window-1:] = pd.Series(low).rolling(window=window, min_periods=window).min().values[window-1:]
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Donchian and 1d indicators
    
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
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        curr_vol_ma = vol_20_aligned[i]
        curr_chop = chop_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_vol > 1.5 * curr_vol_ma
        
        # Regime filter: chop < 38.2 for trending market
        trending_regime = curr_chop < 38.2
        chop_exit = curr_chop > 61.8  # exit when choppy
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper + volume confirm + trending regime
            if (curr_close > curr_upper and 
                vol_confirm and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume confirm + trending regime
            elif (curr_close < curr_lower and 
                  vol_confirm and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR chop regime shifts to choppy
            if (curr_close < curr_lower or chop_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR chop regime shifts to choppy
            if (curr_close > curr_upper or chop_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals