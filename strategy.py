#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout + 1w EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Camarilla H3/L3 levels from weekly data act as strong support/resistance. 
Breaking above H3 with volume and weekly uptrend signals bullish momentum; 
breaking below L3 with volume and weekly downtrend signals bearish momentum.
The weekly EMA50 filter ensures alignment with higher timeframe trend, effective in both bull/bear markets.
ATR-based stoploss limits downside during whipsaws. 12h timeframe targets ~12-25 trades/year to minimize fee drag.
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
    
    # Get 1w data for EMA50 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate H3 and L3 for each 1w bar
    rng = high_1w - low_1w
    h3 = close_1w + 1.1 * rng / 4
    l3 = close_1w - 1.1 * rng / 4
    
    # Align to 12h timeframe (use previous week's levels, so shift by 1)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3, additional_delay_bars=1)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3, additional_delay_bars=1)
    
    # Calculate ATR for stoploss (using 12h data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50 and ATR warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above H3 AND above 1w EMA50 (uptrend filter)
            long_condition = (curr_close > h3_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below L3 AND below 1w EMA50 (downtrend filter)
            short_condition = (curr_close < l3_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Check stoploss: 2.0 * ATR below entry
            if curr_close <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit long: price returns below H3 or trend breaks
            elif curr_close <= h3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Check stoploss: 2.0 * ATR above entry
            if curr_close >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit short: price returns above L3 or trend breaks
            elif curr_close >= l3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0