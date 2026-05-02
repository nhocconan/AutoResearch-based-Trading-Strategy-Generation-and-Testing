#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation + chop regime filter
# Uses 1d HTF for EMA50 to capture long-term trend and reduce false breakouts in ranging markets.
# Donchian(20) from 4h provides proven price channel structure for breakouts.
# Volume confirmation at 1.8x average ensures strong participation while limiting trades (~20-40/year target).
# Choppiness index regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) avoids false signals in sideways markets.
# Discrete sizing 0.25 to balance opportunity and fee drag. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

name = "4h_Donchian20_Breakout_1dEMA50_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian channels (20-period) from 4h timeframe (using prior completed 4h bar)
    # Need at least 21 bars for 20-period lookback + current
    if len(prices) < 21:
        return np.zeros(n)
    
    # Shift by 1 to use only completed bars (no look-ahead)
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    
    # 20-period rolling max/min on prior completed bars
    high_ma = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    donchian_upper = high_ma  # Upper band: 20-period high
    donchian_lower = low_ma   # Lower band: 20-period low
    
    # 1d EMA50 for trend filter (long-term trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging market (avoid breakouts)
    # CHOP < 38.2 = trending market (favor breakouts)
    if len(prices) < 14:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 (no prior close)
    tr[0] = 0
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    atr_14_safe = np.where(atr_14 == 0, 1e-10, atr_14)
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Choppiness Index: 100 * log10(tr_sum / range_14) / log10(14)
    # Avoid division by zero and log of zero
    ratio = np.where(range_14 == 0, 1e-10, tr_sum / range_14)
    # Avoid log of zero or negative
    ratio_safe = np.maximum(ratio, 1e-10)
    chop = 100 * np.log10(ratio_safe) / np.log10(14)
    
    # Regime filter: only trade when CHOP < 38.2 (trending market)
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(100, 20, 50, 14)  # Ensure all indicators are warmed up
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper AND price > 1d EMA50 AND volume spike AND trending regime
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i] and 
                trending_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND price < 1d EMA50 AND volume spike AND trending regime
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i] and 
                  trending_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian lower OR price < 1d EMA50
            if close[i] < donchian_lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian upper OR price > 1d EMA50
            if close[i] > donchian_upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals