#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: On 12h timeframe, Donchian(20) breakouts capture medium-term trend starts.
Combined with 1d EMA34 trend filter and volume spike confirmation, this strategy
enters in direction of higher timeframe trend with momentum. Works in bull via
upside breakouts, bear via downside breakouts. Target: 12-30 trades/year on 12h.
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
    
    # Get 1d data for EMA34 trend filter (MTF - ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR and volume calculations
    start_idx = max(30, 20)  # ATR(14) + volume lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1d_aligned[i]
        atr_val = atr[i]
        
        # Donchian(20) channels: 20-period high/low
        lookback = 20
        if i >= lookback:
            donchian_high = np.max(high[i-lookback:i])
            donchian_low = np.min(low[i-lookback:i])
        else:
            donchian_high = np.max(high[:i])
            donchian_low = np.min(low[:i])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter from 1d EMA34
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        if position == 0:
            # Long: price breaks above Donchian high AND uptrend AND volume spike
            long_condition = curr_close > donchian_high and uptrend and volume_spike
            # Short: price breaks below Donchian low AND downtrend AND volume spike
            short_condition = curr_close < donchian_low and downtrend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below Donchian low
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above Donchian high
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0