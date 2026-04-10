#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + 1w volume spike + chop regime filter
# - Primary: 1d Donchian channel breakout (20-period) for directional bias
# - HTF: 1w volume confirmation (current week volume > 1.5x 20-week MA) + chop regime filter (CHOP < 50 = trending)
# - Long: Price breaks above Donchian upper + volume confirmation + chop regime (trending)
# - Short: Price breaks below Donchian lower + volume confirmation + chop regime (trending)
# - Exit: Opposite Donchian breakout or chop regime shifts to ranging (CHOP > 60)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, volume confirms conviction, chop filter avoids false signals in ranging markets
# - Target: 30-100 trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian Channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if not np.isnan(high_1d[i-lookback+1:i+1]).any() and not np.isnan(low_1d[i-lookback+1:i+1]).any():
            upper[i] = np.max(high_1d[i-lookback+1:i+1])
            lower[i] = np.min(low_1d[i-lookback+1:i+1])
    
    # Calculate 1d Donchian breakout signals
    breakout_up = np.zeros(n, dtype=bool)
    breakout_down = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        if not np.isnan(close_1d[i]) and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
            breakout_up[i] = close_1d[i] > upper[i-1]
            breakout_down[i] = close_1d[i] < lower[i-1]
    
    # Calculate 1w volume moving average (20-period) for volume confirmation
    volume_ma_20_1w = np.full(len(volume_1w), np.nan)
    for i in range(19, len(volume_1w)):
        if not np.isnan(volume_1w[i-19:i+1]).any():
            volume_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Calculate 1w Chopiness Index (CHOP) for regime filter
    chop_lookback = 14
    atr1 = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(np.roll(high_1w, 1) - low_1w),
                                np.abs(np.roll(low_1w, 1) - high_1w)))
    
    # True Range for 1-period
    tr1 = np.maximum(np.maximum(high_1w - low_1w,
                               np.abs(np.roll(high_1w, 1) - low_1w)),
                    np.abs(np.roll(low_1w, 1) - high_1w))
    
    # Sum of TR over chop_lookback period
    sum_tr = np.full(len(tr1), np.nan)
    for i in range(chop_lookback, len(tr1)):
        if not np.isnan(tr1[i-chop_lookback:i]).any():
            sum_tr[i] = np.sum(tr1[i-chop_lookback:i])
    
    # Highest high and lowest low over chop_lookback period
    hh = np.full(len(high_1w), np.nan)
    ll = np.full(len(low_1w), np.nan)
    for i in range(chop_lookback, len(high_1w)):
        if not np.isnan(high_1w[i-chop_lookback:i+1]).any() and not np.isnan(low_1w[i-chop_lookback:i+1]).any():
            hh[i] = np.max(high_1w[i-chop_lookback:i+1])
            ll[i] = np.min(low_1w[i-chop_lookback:i+1])
    
    # Chopiness Index
    chop = np.full(len(high_1w), np.nan)
    for i in range(chop_lookback, len(high_1w)):
        if (not np.isnan(sum_tr[i]) and not np.isnan(hh[i]) and not np.isnan(ll[i]) and 
            hh[i] > ll[i] and sum_tr[i] > 0):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(chop_lookback)
        else:
            chop[i] = np.nan
    
    # Align all HTF indicators to 1d timeframe
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):  # Start after Donchian warmup period
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_20_1w_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period MA
        volume_confirm = volume_1w[-1] > 1.5 * volume_ma_20_1w_aligned[i] if len(volume_1w) > 0 else False
        
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