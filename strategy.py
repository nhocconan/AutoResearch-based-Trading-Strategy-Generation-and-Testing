#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND chop > 61.8 (range regime).
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND chop > 61.8.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-50 trades/year.
# Donchian channels provide structural breakout signals; volume confirms conviction; chop filter avoids whipsaws in strong trends.
# Works in bull (breakouts continue) and bear (breakdowns continue) by trading with momentum in range regimes.

name = "4h_Donchian20_VolumeConfirm_ChopRegime_v1"
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
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(n) * range))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14) for 1d
    atr_1d = np.zeros(len(tr1))
    atr_1d[13] = np.mean(tr1[:14])  # seed
    for i in range(14, len(tr1)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr1[i]) / 14
    
    # Chopiness Index (14)
    chop_1d = np.full_like(close_1d, np.nan)
    lookback = 14
    for i in range(lookback, len(chop_1d)):
        atr_sum = np.sum(atr_1d[i-lookback+1:i+1])
        max_high = np.max(high_1d[i-lookback+1:i+1])
        min_low = np.min(low_1d[i-lookback+1:i+1])
        range_val = max_high - min_low
        if range_val > 0 and atr_sum > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / (np.log10(lookback) * range_val))
        else:
            chop_1d[i] = 50.0  # neutral
    
    # Align 1d chop to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian(20) on 4h
    lookback_dc = 20
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    
    for i in range(lookback_dc-1, len(close)):
        donchian_high[i] = np.max(high[i-lookback_dc+1:i+1])
        donchian_low[i] = np.min(low[i-lookback_dc+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_lookback = 20
    for i in range(vol_lookback-1, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_lookback+1:i+1])
    volume_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_dc, vol_lookback, 30)  # warmup
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_ratio = volume_ratio[i]
        curr_chop = chop_1d_aligned[i]
        
        # Regime filter: only trade in range markets (chop > 61.8)
        in_range = curr_chop > 61.8
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND volume confirmation AND range regime
            if (curr_close > donchian_high[i] and 
                curr_volume_ratio > 1.5 and 
                in_range):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmation AND range regime
            elif (curr_close < donchian_low[i] and 
                  curr_volume_ratio > 1.5 and 
                  in_range):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR chop drops below 38.2 (trend regime)
            if (curr_close < donchian_low[i] or 
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR chop drops below 38.2 (trend regime)
            if (curr_close > donchian_high[i] or 
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals