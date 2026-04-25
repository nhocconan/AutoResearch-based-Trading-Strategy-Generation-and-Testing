#!/usr/bin/env python3
"""
12h Donchian Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. When aligned with 1d EMA34 trend and confirmed by volume spikes, 
this strategy filters false breakouts and works in both bull and bear markets. The 12h timeframe targets 12-37 trades/year 
(50-150 over 4 years) by requiring confluence of trend, breakout, and volume, minimizing fee drag. Uses discrete position sizing 
(0.25) to reduce churn and ATR-based stoploss for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr = np.zeros(n)  # ATR for stoploss
    
    # Calculate ATR (14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_raw = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr = np.concatenate([[np.nan] * 14, atr_raw]) if len(atr_raw) < n else np.concatenate([[np.nan] * 14, atr_raw[:n-14]])
    
    # Start index: need enough for calculations
    start_idx = max(20, 34, 14)  # volume MA, EMA34, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Donchian channel (20) breakout levels
        lookback = 20
        if i < lookback:
            continue
        highest_high = np.max(high[i-lookback:i])
        lowest_low = np.min(low[i-lookback:i])
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Donchian breakout + trend + volume
            # Long: price breaks above Donchian high AND bullish bias AND volume spike
            long_entry = (curr_high > highest_high) and bullish_bias and vol_spike
            # Short: price breaks below Donchian low AND bearish bias AND volume spike
            short_entry = (curr_low < lowest_low) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low (mean reversion) OR loss of bullish bias OR ATR stoploss
            atr_stop = curr_low < (highest_high - 2.5 * atr[i])  # ATR-based stop
            if (curr_low < lowest_low) or (curr_close < ema_1d_aligned[i]) or atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (mean reversion) OR loss of bearish bias OR ATR stoploss
            atr_stop = curr_high > (lowest_low + 2.5 * atr[i])  # ATR-based stop
            if (curr_high > highest_high) or (curr_close > ema_1d_aligned[i]) or atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0