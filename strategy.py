#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum moves. When aligned with
12h EMA50 trend and confirmed by volume spikes, these breakouts have higher reliability.
ATR-based stoploss limits drawdown. Designed for 4h timeframe with tight entry conditions
to achieve 20-50 trades/year. Works in bull (breakouts above upper channel in uptrend)
and bear (breakouts below lower channel in downtrend) by only taking trend-aligned breaks.
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
    
    # Get 12h data for EMA50 trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        atr_val = atr[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > upper_channel) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < lower_channel) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr_val
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below lower Donchian OR price < EMA (trend change) OR stoploss hit
            if (curr_low < lower_channel) or (curr_close < ema_trend) or (curr_close < atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, curr_close - 2.5 * atr_val)
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian OR price > EMA (trend change) OR stoploss hit
            if (curr_high > upper_channel) or (curr_close > ema_trend) or (curr_close > atr_stop):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, curr_close + 2.5 * atr_val)
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0