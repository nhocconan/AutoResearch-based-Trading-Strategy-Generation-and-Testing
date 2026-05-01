#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND ATR(14) > ATR(50) AND volume > 1.5 * volume MA(20)
# Short when: price breaks below Donchian(20) low AND ATR(14) > ATR(50) AND volume > 1.5 * volume MA(20)
# Uses discrete sizing 0.25. ATR regime filter ensures we only trade in sufficient volatility conditions.
# Volume confirmation adds conviction to breakouts. Designed for 12h timeframe to target 12-37 trades/year.

name = "12h_Donchian20_ATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) and ATR(50)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d ATR values to 12h
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Donchian(20) on 12h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: volume > 1.5 * 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for ATR and Donchian
    
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or
            np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_atr_14 = atr_14_1d_aligned[i]
        curr_atr_50 = atr_50_1d_aligned[i]
        curr_vol_thresh = volume_threshold[i]
        
        # ATR regime filter: only trade when short-term ATR > long-term ATR (expanding volatility)
        atr_regime = curr_atr_14 > curr_atr_50
        
        # Volume confirmation
        volume_confirmed = curr_volume > curr_vol_thresh
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND ATR regime AND volume confirmed
            if (curr_close > curr_highest_high and 
                atr_regime and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND ATR regime AND volume confirmed
            elif (curr_close < curr_lowest_low and 
                  atr_regime and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR ATR regime changes
            if (curr_close < curr_lowest_low or 
                not atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR ATR regime changes
            if (curr_close > curr_highest_high or 
                not atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals