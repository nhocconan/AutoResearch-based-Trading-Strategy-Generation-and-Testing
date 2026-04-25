#!/usr/bin/env python3
"""
6h_Irwin_Breakout_1dTrend_VolumeSpike
Hypothesis: Irwin Oscillator (EMA5-EMA34) on 6h with 1d EMA34 trend filter and volume spike confirmation.
Irwin Oscillator > 0 indicates bullish momentum, < 0 bearish. Enter on zero-cross with trend alignment and volume spike.
Targets 15-30 trades/year by requiring confluence of momentum, trend, and volume. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Irwin Oscillator on 6h: EMA(5) - EMA(34)
    close_s = pd.Series(close)
    ema5 = close_s.ewm(span=5, adjust=False, min_periods=5).mean().values
    ema34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    irwin = ema5 - ema34  # >0 bullish, <0 bearish
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (using 6h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Irwin EMA34 (34), 1d EMA34 (34), volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(irwin[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_irwin = irwin[i]
        prev_irwin = irwin[i-1]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals: Irwin zero-cross with trend and volume
            # Long: Irwin crosses above zero with uptrend and volume spike
            long_entry = (prev_irwin <= 0 and curr_irwin > 0) and uptrend and volume_spike[i]
            # Short: Irwin crosses below zero with downtrend and volume spike
            short_entry = (prev_irwin >= 0 and curr_irwin < 0) and downtrend and volume_spike[i]
            
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
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if Irwin crosses below zero (momentum loss) or trend changes
            elif curr_irwin < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if Irwin crosses above zero (momentum loss) or trend changes
            elif curr_irwin > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Irwin_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0