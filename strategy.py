#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian 20-bar breakouts on 4h chart capture strong momentum moves.
Breakouts in direction of daily EMA34 trend with volume confirmation provide edge in both bull and bear markets.
ATR-based stoploss limits drawdown. Designed for 4h timeframe targeting 20-50 trades/year.
Uses discrete position sizing (0.30) to minimize fee churn. Works in bull via breakout continuation
and in bear via mean-reversion from extreme levels when daily trend aligns.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss using vectorized calculation
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
    
    # Calculate 20-period volume MA for volume spike detection using vectorized approach
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Donchian channels (20-period) for breakout signals
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34_1d, ATR, volume MA, and Donchian to propagate
    start_idx = max(34, 14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema34_1d = ema_34_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend (price > 1d EMA34) AND volume spike
            long_condition = (curr_close > upper) and (curr_close > ema34_1d) and volume_spike
            # Short: price breaks below lower Donchian AND downtrend (price < 1d EMA34) AND volume spike
            short_condition = (curr_close < lower) and (curr_close < ema34_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below lower Donchian (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above upper Donchian (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0