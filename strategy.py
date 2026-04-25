#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian breakouts capture momentum. Align with 1d EMA34 trend to filter counter-trend moves.
Volume spike confirms participation. Chop filter avoids whipsaws in ranging markets.
Works in bull via buying upper band breakouts, bear via selling lower band breakdowns.
Uses discrete position sizing (0.25) to control drawdown. Target: 20-50 trades/year on 4h.
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
    
    # Get 1d data for EMA34 trend filter and Chop filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Chop filter (EWMA of True Range / ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    # True Range of 1d
    tr_1d_latest = pd.concat([
        pd.Series(high_1d).diff().abs(),
        (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs(),
        (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    ], axis=1).max(axis=1)
    # Chop = 100 * log10(sum(TR14) / (log10(14) * (HH14 - LL14)))
    sum_tr_14 = pd.Series(tr_1d_latest).rolling(window=14, min_periods=14).sum()
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop_raw = 100 * np.log10(sum_tr_14 / (np.log10(14) * (hh_14 - ll_14)))
    chop_values = chop_raw.fillna(50).values  # fill NaN with 50 (neutral)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate ATR(14) for stoploss on 4h
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and EMA34 to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Donchian(20): highest high and lowest low of past 20 completed periods
        if i >= 20:
            highest_20 = np.max(high[i-20:i])
            lowest_20 = np.min(low[i-20:i])
        else:
            highest_20 = np.max(high[max(0, i-19):i+1])
            lowest_20 = np.min(low[max(0, i-19):i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Chop filter: only trade when market is trending (Chop < 38.2) or extreme ranging (Chop > 61.8) for mean reversion?
        # For breakout strategy, we want trending markets: Chop < 38.2
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Long: break above Donchian upper band AND uptrend AND volume spike AND trending regime
            long_condition = curr_high > highest_20 and curr_close > ema_34 and volume_spike and trending_regime
            # Short: break below Donchian lower band AND downtrend AND volume spike AND trending regime
            short_condition = curr_low < lowest_20 and curr_close < ema_34 and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA34
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA34
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0