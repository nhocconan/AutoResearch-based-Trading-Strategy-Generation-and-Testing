#!/usr/bin/env python3
"""
1h ATR Breakout with 4h EMA Trend Filter and Volume Spike
Hypothesis: Breakouts above/below ATR-based channels during strong 4h trends with volume confirmation capture sustained moves. 4h EMA filter ensures we trade with the higher timeframe trend, reducing whipsaws in both bull and bear markets. Volume spike confirms institutional participation. Designed for 1h timeframe with tight entry conditions to limit trades to 15-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter and ATR calculation (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 4h
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14_4h = np.full(len(df_4h), np.nan)
    for i in range(14, len(tr)):
        atr_14_4h[i+1] = np.mean(tr[i-13:i+1])
    
    # Get 1d data for session-independent daily ATR (for channel width)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate ATR(10) on 1d for breakout channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1_1d, tr2_1d), tr3_1d)
    atr_10_1d = np.full(len(df_1d), np.nan)
    for i in range(10, len(tr_1d)):
        atr_10_1d[i+1] = np.mean(tr_1d[i-9:i+1])
    
    # Align HTF indicators to 1h
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Calculate 1h ATR(14) for dynamic position sizing (optional, using fixed size)
    # Calculate 20-period volume MA on 1h for volume spike
    vol_ma_20_1h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR and volume MA
    start_idx = max(30, 20)  # ATR needs 30, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(atr_10_1d_aligned[i]) or
            np.isnan(vol_ma_20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_20_4h_aligned[i]
        atr_4h = atr_14_4h_aligned[i]
        atr_1d = atr_10_1d_aligned[i]
        vol_ma = vol_ma_20_1h[i]
        
        # Dynamic breakout channels based on 1d ATR (adjusts to volatility)
        # Upper channel: previous close + 0.7 * 1d ATR
        # Lower channel: previous close - 0.7 * 1d ATR
        prev_close = close[i-1]
        upper_channel = prev_close + 0.7 * atr_1d
        lower_channel = prev_close - 0.7 * atr_1d
        
        # Volume confirmation: current volume > 2.5 * 20-period MA
        volume_confirm = curr_volume > 2.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper channel, above 4h EMA, volume confirmation
            long_entry = (curr_close > upper_channel and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower channel, below 4h EMA, volume confirmation
            short_entry = (curr_close < lower_channel and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below 4h EMA OR hits trailing stop (2 * 4h ATR from entry)
            if curr_close < ema_trend or curr_close < entry_price - 2.0 * atr_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price rises above 4h EMA OR hits trailing stop (2 * 4h ATR from entry)
            if curr_close > ema_trend or curr_close > entry_price + 2.0 * atr_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ATR_Breakout_4hEMA20_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0