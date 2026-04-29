#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses 1w Camarilla pivot levels (R4/S4) for trend bias and breakout confirmation
# Donchian(20) from previous 20 6h bars act as entry/exit levels
# Volume spike (2.0x 20-period average) confirms breakout validity
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Weekly pivot provides structural bias that works in both bull and bear regimes

name = "6h_Donchian_Breakout_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Align weekly levels to 6h timeframe (completed weekly bars only)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for ATR
    
    for i in range(start_idx, n):
        # Need at least 20 previous bars for Donchian calculation
        if i < 20:
            signals[i] = 0.0
            continue
            
        # Calculate Donchian levels from previous 20 bars (excluding current)
        donchian_high = np.max(high[i-20:i])
        donchian_low = np.min(low[i-20:i])
        
        curr_close = close[i]
        curr_r4 = r4_1w_aligned[i]
        curr_s4 = s4_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below Donchian low OR price below weekly S4 OR stoploss hit
            if curr_close < donchian_low or curr_close < curr_s4 or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above Donchian high OR price above weekly R4 OR stoploss hit
            if curr_close > donchian_high or curr_close > curr_r4 or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > weekly R4 AND volume spike
            if curr_close > donchian_high and curr_close > curr_r4 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND price < weekly S4 AND volume spike
            elif curr_close < donchian_low and curr_close < curr_s4 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals