#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + chop regime filter
# Donchian breakouts capture strong directional moves in both bull and bear markets
# Volume confirmation ensures breakouts have institutional participation
# Chop regime filter (Choppiness Index) avoids false breakouts in ranging markets
# In trending regimes (CHOP < 38.2): trade breakouts in direction of trend
# In ranging regimes (CHOP > 61.8): fade Donchian touches (mean reversion)
# Position size 0.25 to limit drawdown during 2022-style crashes
# Target: 75-200 total trades over 4 years (19-50/year) for optimal fee drag balance

name = "4h_1d_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # Sum of TR over 14 periods
    tr_sum_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        tr_sum_14[i] = np.sum(tr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full(len(df_1d), np.nan)
    ll_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        hh_14[i] = np.max(high_1d[i-13:i+1])
        ll_14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if hh_14[i] > ll_14[i]:
            chop_1d[i] = 100 * np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Align 1d data to 4h timeframe
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_1d_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Donchian breakout signals
    breakout_up = close > highest_high  # Close above upper band
    breakout_down = close < lowest_low  # Close below lower band
    
    # Volume confirmation: current 4h volume > 1.5x 20-period 1d average volume (aligned)
    volume_confirm = volume > (1.5 * vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20_4h[i]) or 
            np.isnan(chop_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_1d_4h[i]
        vol_conf = volume_confirm[i]
        break_up = breakout_up[i]
        break_down = breakout_down[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop < 38.2:  # Trending regime
                # Exit when price closes below midpoint (trailing stop)
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] < midpoint:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price returns to opposite Donchian band
                if close[i] >= lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop < 38.2:  # Trending regime
                # Exit when price closes above midpoint
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] > midpoint:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price returns to opposite Donchian band
                if close[i] <= highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if chop < 38.2:  # Trending regime - trade breakouts
                if break_up and vol_conf:
                    position = 1
                    signals[i] = 0.25
                elif break_down and vol_conf:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime (CHOP > 38.2) - fade extreme touches
                # Only trade if chop indicates ranging (above 38.2)
                if chop > 38.2:
                    # Long when price touches lower band with volume confirmation
                    if close[i] <= lowest_low[i] and vol_conf:
                        position = 1
                        signals[i] = 0.25
                    # Short when price touches upper band with volume confirmation
                    elif close[i] >= highest_high[i] and vol_conf:
                        position = -1
                        signals[i] = -0.25
    
    return signals