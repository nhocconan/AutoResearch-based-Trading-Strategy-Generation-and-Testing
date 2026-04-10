#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d volume spike + chop regime filter
# - Primary: 4h Donchian channel breakout (20-period) for directional bias
# - HTF: 1d volume confirmation (current day volume > 1.5x 20-day MA) + chop regime filter (CHOP < 50 = trending)
# - Long: Price breaks above Donchian upper + volume confirmation + chop regime (trending)
# - Short: Price breaks below Donchian lower + volume confirmation + chop regime (trending)
# - Exit: Opposite Donchian breakout or chop regime shifts to ranging (CHOP > 60)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, volume confirms conviction, chop filter avoids false signals in ranging markets
# - Target: 75-200 trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if not np.isnan(high_4h[i-lookback+1:i+1]).any() and not np.isnan(low_4h[i-lookback+1:i+1]).any():
            upper[i] = np.max(high_4h[i-lookback+1:i+1])
            lower[i] = np.min(low_4h[i-lookback+1:i+1])
    
    # Calculate 4h Donchian breakout signals
    breakout_up = np.zeros(n, dtype=bool)
    breakout_down = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        if not np.isnan(close_4h[i]) and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
            breakout_up[i] = close_4h[i] > upper[i-1]
            breakout_down[i] = close_4h[i] < lower[i-1]
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    chop_lookback = 14
    atr1 = np.maximum(high_1d - low_1d, 
                      np.maximum(np.abs(np.roll(high_1d, 1) - low_1d),
                                np.abs(np.roll(low_1d, 1) - high_1d)))
    
    # True Range for 1-period
    tr1 = np.maximum(np.maximum(high_1d - low_1d,
                               np.abs(np.roll(high_1d, 1) - low_1d)),
                    np.abs(np.roll(low_1d, 1) - high_1d))
    
    # Sum of TR over chop_lookback period
    sum_tr = np.full(len(tr1), np.nan)
    for i in range(chop_lookback, len(tr1)):
        if not np.isnan(tr1[i-chop_lookback:i]).any():
            sum_tr[i] = np.sum(tr1[i-chop_lookback:i])
    
    # Highest high and lowest low over chop_lookback period
    hh = np.full(len(high_1d), np.nan)
    ll = np.full(len(low_1d), np.nan)
    for i in range(chop_lookback, len(high_1d)):
        if not np.isnan(high_1d[i-chop_lookback:i+1]).any() and not np.isnan(low_1d[i-chop_lookback:i+1]).any():
            hh[i] = np.max(high_1d[i-chop_lookback:i+1])
            ll[i] = np.min(low_1d[i-chop_lookback:i+1])
    
    # Chopiness Index
    chop = np.full(len(high_1d), np.nan)
    for i in range(chop_lookback, len(high_1d)):
        if (not np.isnan(sum_tr[i]) and not np.isnan(hh[i]) and not np.isnan(ll[i]) and 
            hh[i] > ll[i] and sum_tr[i] > 0):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(chop_lookback)
        else:
            chop[i] = np.nan
    
    # Align all HTF indicators to 4h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after Donchian warmup period
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 4h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP < 50 indicates trending market (avoid ranging)
        regime_confirm = chop_aligned[i] < 50.0
        
        # Donchian breakout signals
        donchian_up = breakout_up[i]
        donchian_down = breakout_down[i]
        
        # Exit conditions: Opposite Donchian breakout OR chop regime shifts to ranging (CHOP > 60)
        exit_long = donchian_down or (chop_aligned[i] > 60.0)
        exit_short = donchian_up or (chop_aligned[i] > 60.0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian breakout up + volume confirmation + trending regime
            if donchian_up and volume_confirm and regime_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakout down + volume confirmation + trending regime
            elif donchian_down and volume_confirm and regime_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Donchian breakout OR chop regime shifts to ranging
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals