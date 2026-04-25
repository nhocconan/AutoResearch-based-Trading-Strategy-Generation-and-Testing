#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d EMA34 Trend and Volume Spike + ATR Stop
Hypothesis: Camarilla H3/L3 levels act as key intraday pivot points. Breakouts above H3 with volume confirmation and 1d EMA34 uptrend signal strong momentum longs. Breakdowns below L3 with volume confirmation and 1d EMA34 downtrend signal strong momentum shorts. Using 1d EMA34 as HTF trend filter ensures alignment with long-term trend, reducing false signals. Volume spike confirms participation. ATR-based stoploss limits downside. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year on 4h.
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
    
    # Calculate ATR for stoploss (using 14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # 4h timeframe: 6 bars per day
    lookback = 6
    
    # Calculate rolling max/min/close for prior day (excluding current bar)
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
    start_idx = max(lookback, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
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
            # Exit: price falls below L3 (mean reversion) OR trend change (price < EMA) OR ATR stoploss
            atr_stop = entry_price - 2.0 * atr_val
            if (curr_close < L3[i]) or (curr_close < ema_trend) or (curr_close < atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (mean reversion) OR trend change (price > EMA) OR ATR stoploss
            atr_stop = entry_price + 2.0 * atr_val
            if (curr_close > H3[i]) or (curr_close > ema_trend) or (curr_close > atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATR_SL"
timeframe = "4h"
leverage = 1.0