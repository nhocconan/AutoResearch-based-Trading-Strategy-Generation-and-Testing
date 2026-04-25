#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Pullback_1dTrend_VolumeConfirm
Hypothesis: On 12h timeframe, enter long on pullback to L3 in uptrend (price > 1d EMA34) and short on pullback to H3 in downtrend (price < 1d EMA34), with volume confirmation and ATR stoploss. Uses 1d Camarilla levels from prior day, 1d EMA34 for trend, and volume > 1.5x 20-period mean. Designed for low trade frequency (12-37/year) to minimize fee drag. Works in bull via pullback longs and in bear via pullback shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H3 = prev_close + 1.125 * prev_range
    L3 = prev_close - 1.125 * prev_range
    
    # Align 1d pivot levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (34 for EMA, 20 for vol MA, 14 for ATR)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals
            # Long: pullback to L3 in uptrend with volume spike
            long_entry = (curr_close <= L3_aligned[i]) and (curr_close > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: pullback to H3 in downtrend with volume spike
            short_entry = (curr_close >= H3_aligned[i]) and (curr_close < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on stoploss or mean reversion
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit on reversion to mean (EMA34) or opposite Camarilla level
            elif curr_close >= ema_34_1d_aligned[i] or curr_close >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or mean reversion
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit on reversion to mean (EMA34) or opposite Camarilla level
            elif curr_close <= ema_34_1d_aligned[i] or curr_close <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Pullback_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0