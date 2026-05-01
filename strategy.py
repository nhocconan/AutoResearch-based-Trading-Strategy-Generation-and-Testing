#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 1d chop regime filter.
# Long when: price breaks above Donchian(20) high AND 12h volume > 1.5x 20-period average AND 1d chop < 61.8 (trending)
# Short when: price breaks below Donchian(20) low AND 12h volume > 1.5x 20-period average AND 1d chop < 61.8 (trending)
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-50 trades/year.
# Donchian channels provide objective breakout levels, volume confirms conviction, chop filter avoids ranging markets.
# Works in bull (breakouts continuation) and bear (breakdowns continuation) by trading with the trend.

name = "4h_Donchian20_VolumeConfirm_12h_1dChopRegime_v1"
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
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 1d chop regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(atr/14) / log10(highest-high - lowest-low) / 14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high - lowest_low
    chop_raw = 100 * np.log10(atr_sum / (range_14 * 14)) / np.log10(10)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
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
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_ma = vol_ma_12h_aligned[i]
        curr_chop = chop_aligned[i]
        curr_donch_high = highest_high_20[i]
        curr_donch_low = lowest_low_20[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        # Get current 12h volume (need to align)
        idx_12h = i // (12*4)  # 12h = 48 * 15m bars, but we're on 4h so 12h = 3 * 4h bars
        if idx_12h < len(df_12h):
            curr_vol_12h = df_12h['volume'].iloc[idx_12h]
        else:
            curr_vol_12h = 0
        
        volume_confirmed = curr_vol_12h > (curr_vol_ma * 1.5)
        
        # Chop regime: trending market (chop < 61.8)
        trending_regime = curr_chop < 61.8
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND volume confirmed AND trending regime
            if (curr_close > curr_donch_high and 
                volume_confirmed and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND volume confirmed AND trending regime
            elif (curr_close < curr_donch_low and 
                  volume_confirmed and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop becomes ranging (> 61.8)
            if (curr_close < curr_donch_low or 
                curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop becomes ranging (> 61.8)
            if (curr_close > curr_donch_high or 
                curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals