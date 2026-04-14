#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h ADX + 1d Bollinger Bands squeeze + 4h momentum
# ADX(14) > 25 indicates trending market, BB width < 20th percentile indicates low volatility squeeze
# When ADX confirms trend AND BB squeeze breaks out, we enter in breakout direction
# Works in bull/bear as it captures volatility expansion after squeeze in trending markets
# Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for ADX
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX (14)
    adx_len = 14
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        first_avg = np.nansum(arr[1:period+1])  # skip first nan
        result[period] = first_avg
        # Wilder smoothing
        for i in range(period+1, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smooth = smooth_wilder(tr, adx_len)
    dm_plus_smooth = smooth_wilder(dm_plus, adx_len)
    dm_minus_smooth = smooth_wilder(dm_minus, adx_len)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= adx_len:
        # First ADX is average of first adx_len DX values
        first_adx = np.nanmean(dx[adx_len:2*adx_len])
        adx[2*adx_len-1] = first_adx
        # Wilder smoothing for ADX
        for i in range(2*adx_len, len(dx)):
            adx[i] = (adx[i-1] * (adx_len-1) + dx[i]) / adx_len
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Load 1d data ONCE for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_len = 20
    bb_mult = 2.0
    bb_src = df_1d['close'].values
    
    # Basis (SMA)
    basis = pd.Series(bb_src).rolling(window=bb_len, min_periods=bb_len).mean().values
    # Deviation
    dev = bb_mult * pd.Series(bb_src).rolling(window=bb_len, min_periods=bb_len).std().values
    # Upper and Lower bands
    upper = basis + dev
    lower = basis - dev
    # Bandwidth (normalized by basis)
    bb_width = (upper - lower) / basis
    bb_width = np.where(basis == 0, 0, bb_width)
    
    # Align BB width to 4h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 2*adx_len, bb_len)  # Need enough for ADX and BB
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Squeeze detection: BB width below 20th percentile of last 50 periods
        squeeze = False
        if i >= 50:
            bb_width_slice = bb_width_aligned[max(0, i-50):i]
            if len(bb_width_slice) > 0:
                # Filter out NaN values
                valid_widths = bb_width_slice[~np.isnan(bb_width_slice)]
                if len(valid_widths) > 0:
                    sorted_widths = np.sort(valid_widths)
                    current_width = bb_width_aligned[i]
                    rank = np.searchsorted(sorted_widths, current_width, side='left')
                    percentile = (rank / len(sorted_widths)) * 100
                    squeeze = percentile <= 20  # Low volatility squeeze
        
        # Momentum confirmation: price > open for bullish, price < open for bearish
        bullish_mom = price > prices['open'].iloc[i]
        bearish_mom = price < prices['open'].iloc[i]
        
        if position == 0:
            # Enter long: trending + squeeze breakout + bullish momentum
            if trending and squeeze and bullish_mom:
                position = 1
                signals[i] = position_size
            # Enter short: trending + squeeze breakout + bearish momentum
            elif trending and squeeze and bearish_mom:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens (ADX < 20) OR momentum reverses
            if adx_aligned[i] < 20 or not bullish_mom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakens OR momentum reverses
            if adx_aligned[i] < 20 or not bearish_mom:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hADX_1dBBwidth_Momentum_v1"
timeframe = "4h"
leverage = 1.0