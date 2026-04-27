#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation.
# In high chop (range) markets: fade Donchian breaks (mean reversion).
# In low chop (trend) markets: follow Donchian breaks (trend continuation).
# Volume filter ensures breakouts have conviction. Designed for 20-40 trades/year.
# Works in bull/bear by adapting to regime. Uses 1d Choppiness Index for regime detection.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM for 1d (Wilder's smoothing)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            first_avg = np.nansum(data[:period])
            result[period-1] = first_avg
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Choppiness Index: higher = more choppy/range, lower = more trending
    # CHOP = 100 * log10(sum(ATR)/ (HHV - LLV)) / log10(period)
    def choppiness_index(high_arr, low_arr, close_arr, period):
        atr = wilder_smooth(
            np.maximum(
                np.maximum(high_arr[1:] - low_arr[1:],
                           np.abs(high_arr[1:] - close_arr[:-1])),
                np.abs(low_arr[1:] - close_arr[:-1])
            ), period)
        hhv = np.full_like(high_arr, np.nan)
        llv = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            hhv[i] = np.max(high_arr[i-period+1:i+1])
            llv[i] = np.min(low_arr[i-period+1:i+1])
        sum_atr = np.full_like(high_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        chop = np.full_like(high_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            if hhv[i] != llv[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (hhv[i] - llv[i])) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian channels (20-period) on 4h
    def donchian_channels(high_arr, low_arr, period):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    donchian_high, donchian_low = donchian_channels(high, low, 20)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic:
        # High chop (>61.8) = range -> fade breaks
        # Low chop (<38.2) = trend -> follow breaks
        if chop_1d_aligned[i] > 61.8:  # Range regime
            # Fade Donchian breaks: short at upper band, long at lower band
            if close[i] > donchian_high[i] and volume_filter[i]:
                signals[i] = -0.25  # Short
                position = -1
            elif close[i] < donchian_low[i] and volume_filter[i]:
                signals[i] = 0.25   # Long
                position = 1
            else:
                # Hold position or flat
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        elif chop_1d_aligned[i] < 38.2:  # Trend regime
            # Follow Donchian breaks: long at upper band, short at lower band
            if close[i] > donchian_high[i] and volume_filter[i]:
                signals[i] = 0.25   # Long
                position = 1
            elif close[i] < donchian_low[i] and volume_filter[i]:
                signals[i] = -0.25  # Short
                position = -1
            else:
                # Hold position or flat
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Neutral chop (38.2-61.8) = no clear regime
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_ChoppinessRegime_Donchian20_VolumeFilter"
timeframe = "4h"
leverage = 1.0