#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA50 ensures alignment with daily trend
# Volume spike confirms institutional participation. ATR-based stoploss manages risk.
# Works in bull markets via breakout continuation and in bear markets via breakdown shorts.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 (trend filter)
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for stoploss and volume filter
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(50, 20)  # warmup for 1d EMA50, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_50 = ema_50_1d_aligned[i]
        curr_atr = atr_14_1d_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian + above 1d EMA50 + volume spike
            if curr_close > curr_upper and curr_close > curr_ema_50 and curr_volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.0 * curr_atr
            # Short: price breaks below lower Donchian + below 1d EMA50 + volume spike
            elif curr_close < curr_lower and curr_close < curr_ema_50 and curr_volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.0 * curr_atr
        
        elif position == 1:  # Long position
            # Trail stop: move stop up as price increases
            atr_stop = max(atr_stop, curr_close - 2.0 * curr_atr)
            # Exit if price hits trailing stop or breaks below 1d EMA50
            if curr_low <= atr_stop or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Trail stop: move stop down as price decreases
            atr_stop = min(atr_stop, curr_close + 2.0 * curr_atr)
            # Exit if price hits trailing stop or breaks above 1d EMA50
            if curr_high >= atr_stop or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals