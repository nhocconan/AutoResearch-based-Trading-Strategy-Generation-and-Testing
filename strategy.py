#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d Supertrend(ATR=10,mult=3) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND Supertrend=uptrend AND volume > 1.5x 20-period volume median.
# Short when price breaks below Donchian lower band AND Supertrend=downtrend AND volume > 1.5x 20-period volume median.
# Uses discrete sizing 0.25. ATR(10) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian provides robust price channel structure; Supertrend filters for trend alignment on HTF;
# volume confirmation ensures breakout conviction. Works in bull/bear via trend filter.
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years).

name = "4h_Donchian20_Breakout_1dSupertrend_Volume_v1"
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
    
    # Calculate ATR(10) for Supertrend and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian channels (20-period, using prior bar's data to avoid look-ahead)
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Supertrend(ATR=10,mult=3) trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Supertrend calculation on 1d data
    hl2_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values[:len(df_1d)]  # Use same ATR calculation
    upper_band_1d = hl2_1d + (3 * atr_1d)
    lower_band_1d = hl2_1d - (3 * atr_1d)
    
    supertrend_1d = np.full(len(df_1d), np.nan, dtype=float)
    direction_1d = np.full(len(df_1d), np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_1d)):
        if np.isnan(upper_band_1d[i]) or np.isnan(lower_band_1d[i]) or np.isnan(close_1d := df_1d['close'].values[i]):
            continue
            
        if i == 1:
            supertrend_1d[i] = upper_band_1d[i]
            direction_1d[i] = 1
        else:
            if supertrend_1d[i-1] == upper_band_1d[i-1]:
                supertrend_1d[i] = upper_band_1d[i] if close_1d <= upper_band_1d[i] else lower_band_1d[i]
                direction_1d[i] = -1 if close_1d <= upper_band_1d[i] else 1
            else:
                supertrend_1d[i] = lower_band_1d[i] if close_1d >= lower_band_1d[i] else upper_band_1d[i]
                direction_1d[i] = 1 if close_1d >= lower_band_1d[i] else -1
    
    # Align Supertrend direction to 4h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and Supertrend
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: Supertrend direction
        uptrend = supertrend_dir_aligned[i] > 0
        downtrend = supertrend_dir_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian upper AND uptrend AND volume spike
            if curr_close > donchian_upper[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Donchian lower AND downtrend AND volume spike
            elif curr_close < donchian_lower[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower OR trend turns down
            elif curr_close < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper OR trend turns up
            elif curr_close > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals