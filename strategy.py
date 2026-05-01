#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator trend filter with 1w/1d HTF confluence and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies strong trends via smoothed moving averages
# Only trade in direction of weekly trend (1w EMA50) to avoid counter-trend whipsaws
# Requires 1d ADX > 25 for trending regime filter + volume > 1.5x 20-period EMA for confirmation
# Designed for very low frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Alligator excels in strong trends (both bull and bear) while avoiding choppy markets

name = "12h_WilliamsAlligator_1wEMA50_1dADX_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for weekly trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d HTF data for ADX regime filter and Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for trending regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period]) if not np.all(np.isnan(data[1:period])) else np.nan
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 1d data (smoothed medians)
    def smoothed_median(data, period):
        # Smoothed median = SMMA of median price (HL/2)
        median_price = (high_1d + low_1d) / 2
        result = np.full_like(median_price, np.nan)
        if len(median_price) < period:
            return result
        # First value
        result[period-1] = np.nanmean(median_price[1:period]) if not np.all(np.isnan(median_price[1:period])) else np.nan
        # Subsequent values: SMMA
        for i in range(period, len(median_price)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + median_price[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_1d = smoothed_median((high_1d + low_1d) / 2, 13)
    teeth_1d = smoothed_median((high_1d + low_1d) / 2, 8)
    lips_1d = smoothed_median((high_1d + low_1d) / 2, 5)
    
    # Apply smoothing offsets (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient 1w/1d data
    start_idx = 70
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from weekly EMA50: long above, short below
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        # Alligator signals: Lips > Teeth > Jaw = bullish alignment, Lips < Teeth < Jaw = bearish
        bullish_alligator = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
        bearish_alligator = (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i])
        
        # Regime filter: ADX > 25 for trending market
        trending_regime = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and bullish_alligator and trending_regime:
                # Long: Alligator bullish alignment + weekly trend + volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias and bearish_alligator and trending_regime:
                # Short: Alligator bearish alignment + weekly trend + volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop or counter-trend
        
        elif position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR weekly trend breaks OR ADX weakens
            if (not bullish_alligator) or (close[i] <= ema_50_1w_aligned[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR weekly trend breaks OR ADX weakens
            if (not bearish_alligator) or (close[i] >= ema_50_1w_aligned[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals