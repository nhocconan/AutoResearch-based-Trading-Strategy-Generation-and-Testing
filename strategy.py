#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Choppiness Index (14) regime filter with Weekly Donchian(20) breakout.
# Uses weekly trend context and daily range detection to avoid whipsaws.
# In trending weeks (weekly CHOP < 38.2): daily breakout above/below daily Donchian(20) with volume spike.
# In ranging weeks (weekly CHOP > 61.8): daily mean reversion at daily Donchian boundaries.
# Designed for low trade frequency (<25/year) with regime adaptation for bull/bear markets.
# Target assets: BTC/ETH/SOL with focus on avoiding false breakouts in chop.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for regime filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for Weekly Choppiness Index
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate smoothed TR, +DM, -DM for ADX components
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with 14-period Wilder's smoothing (equivalent to EMA with alpha=1/14)
    tr_smooth = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initialize first values
    tr_smooth[13] = tr[1:14].sum()
    plus_dm_smooth[13] = plus_dm[1:14].sum()
    minus_dm_smooth[13] = minus_dm[1:14].sum()
    
    # Wilder smoothing: new = old - (old/14) + current
    for i in range(14, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/14) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/14) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/14) + minus_dm[i]
    
    # Avoid division by zero
    tr_smooth = np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    
    # Smooth DX for ADX
    adx = np.zeros_like(dx)
    adx[27] = dx[14:28].mean()  # First ADX at index 27 (14+14-1)
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Weekly Choppiness Index: higher = more ranging
    atr_sum = np.zeros_like(tr)
    atr_sum[13] = tr[1:14].sum()
    for i in range(14, len(tr)):
        atr_sum[i] = atr_sum[i-1] - (atr_sum[i-1]/14) + tr[i]
    
    highest_high = np.zeros_like(high_1w)
    lowest_low = np.zeros_like(low_1w)
    highest_high[13] = high_1w[1:14].max()
    lowest_low[13] = low_1w[1:14].min()
    for i in range(14, len(high_1w)):
        highest_high[i] = max(highest_high[i-1], high_1w[i])
        lowest_low[i] = min(lowest_low[i-1], low_1w[i])
    
    # Calculate Choppiness Index
    chop = np.full_like(tr, 50.0)  # Default to neutral
    valid = (highest_high - lowest_low) > 0
    chop[valid] = 100 * np.log10(atr_sum[valid] / (highest_high[valid] - lowest_low[valid])) / np.log10(14)
    
    # Align weekly Choppiness to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Load daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup for all indicators
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume_1d[i]  # Use daily volume aligned to intraday
        vol_ma = vol_ma_20[i]
        chop_val = chop_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        # Volume filter: current volume > 1.5 * 20-period average (strict to reduce trades)
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine weekly regime
            is_trending = chop_val < 38.2  # Trending week
            is_ranging = chop_val > 61.8   # Ranging week
            
            if is_trending:
                # Trending week: breakout in direction of momentum
                if price > upper and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif price < lower and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging week: mean reversion at boundaries
                if price <= lower and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif price >= upper and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: reversion to mean or opposite touch
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on retracement to midpoint or touch of upper band
                mid = (upper + lower) / 2
                if price < mid or price >= upper:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on retracement to midpoint or touch of lower band
                mid = (upper + lower) / 2
                if price > mid or price <= lower:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyChop_Donchian_Breakout"
timeframe = "1d"
leverage = 1.0