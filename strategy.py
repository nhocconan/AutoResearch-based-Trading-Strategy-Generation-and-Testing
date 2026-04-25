#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum moves.
Breakouts in direction of 1d EMA34 trend with volume confirmation work in both bull and bear markets.
In bull markets: breakouts above upper band with uptrend capture continuation.
In bear markets: breakouts below lower band with downtrend capture momentum.
Volume spike filters false breakouts, ATR stoploss manages risk during choppy periods.
Designed for 12h timeframe targeting 12-37 trades/year with discrete position sizing (0.25).
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 12h data
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate Donchian(20) channels on 12h data
    if len(close) >= 20:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(n):
            start_idx = max(0, i - 19)
            upper[i] = np.max(high[start_idx:i+1])
            lower[i] = np.min(low[start_idx:i+1])
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, Donchian, and volume MA to propagate
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1d = ema_34_1d_aligned[i]
        upper_band = upper[i]
        lower_band = lower[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper band AND uptrend (price > 1d EMA34) AND volume spike
            long_condition = (curr_close > upper_band) and (curr_close > ema34_1d) and volume_spike
            # Short: price breaks below lower band AND downtrend (price < 1d EMA34) AND volume spike
            short_condition = (curr_close < lower_band) and (curr_close < ema34_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below lower band (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above upper band (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0