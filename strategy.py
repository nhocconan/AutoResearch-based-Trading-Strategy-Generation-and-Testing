#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d EMA34 Trend and Volume Spike + ATR Stop
Hypothesis: Camarilla H3/L3 levels from prior day act as key support/resistance. 
Breakouts with volume confirmation and alignment with 1d EMA34 trend capture momentum. 
ATR-based stoploss limits downside. Discrete sizing (0.0, ±0.25) reduces fee churn. 
Designed to work in both bull and bear markets by following the 1d trend filter.
Target: 20-50 trades/year on 4h.
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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get ATR for stoploss (using 1d ATR aligned to 4h)
    if len(df_1d) < 14:
        atr_1d = np.zeros(len(df_1d))
    else:
        # Calculate True Range for 1d
        tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
        tr2 = np.abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
        tr3 = np.abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla levels from prior day's OHLC
    # 4h timeframe: 6 bars per day
    lookback = 6
    
    # Prior day's OHLC (excluding current bar)
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla levels
    diff = prev_high - prev_low
    H3 = prev_close + diff * 1.1 / 4
    L3 = prev_close - diff * 1.1 / 4
    
    # Volume confirmation: current volume > 1.5 * 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        atr = atr_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > H3[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < L3[i]) and vol_spike and (curr_close < ema_trend)
            
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
            # Long position management
            # Exit: ATR stoploss OR mean reversion (price < L3) OR trend change (price < EMA)
            stop_price = entry_price - 2.5 * atr
            if (curr_close < stop_price) or (curr_close < L3[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: ATR stoploss OR mean reversion (price > H3) OR trend change (price > EMA)
            stop_price = entry_price + 2.5 * atr
            if (curr_close > stop_price) or (curr_close > H3[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATR_SL"
timeframe = "4h"
leverage = 1.0