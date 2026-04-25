#!/usr/bin/env python3
"""
12h Donchian Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian(20) breakouts on 12h capture medium-term trends, with 1d EMA34 filtering direction.
Volume spike confirms institutional participation, chop filter avoids whipsaws in ranging markets.
Works in bull via buying upper band breakouts, bear via selling lower band breakdowns.
Uses discrete position sizing (0.25) to control drawdown. Target: 12-37 trades/year on 12h.
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
    
    # Get 1d data for EMA34 trend filter and Donchian calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Donchian channels (20-period)
    if len(df_1d) >= 20:
        donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
        donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
    
    # Calculate ATR(14) for stoploss on 12h data
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Choppiness Index(14) on 1d for regime filter
    if len(df_1d) >= 14:
        # True Range
        tr1_1d = pd.Series(df_1d['high']).diff().abs()
        tr2_1d = (pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift()).abs()
        tr3_1d = (pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift()).abs()
        tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
        atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
        
        # Max High - Min Low over 14 periods
        max_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
        min_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
        range_1d = max_high_1d - min_low_1d
        
        # Chop = 100 * log10(sum(atr14) / (log14 * (max_high - min_low)))
        sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        chop = 100 * np.log10(sum_atr_1d / (np.log10(14) * range_1d))
        chop = np.where(range_1d == 0, 50, chop)  # avoid division by zero
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # neutral chop
    
    # Pre-compute volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i >= 0:
            vol_ma_20[i] = np.mean(volume[:i+1])
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1d_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        atr_val = atr[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper band AND uptrend AND low chop AND volume spike
            long_condition = (curr_high > upper_band and 
                             curr_close > ema_34 and 
                             chop_val < 61.8 and 
                             vol_spike)
            # Short: break below lower band AND downtrend AND low chop AND volume spike
            short_condition = (curr_low < lower_band and 
                              curr_close < ema_34 and 
                              chop_val < 61.8 and 
                              vol_spike)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price falls below EMA34 or chop > 61.8
            if (curr_close <= entry_price - 2.5 * atr_val or 
                curr_close < ema_34 or 
                chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price rises above EMA34 or chop > 61.8
            if (curr_close >= entry_price + 2.5 * atr_val or 
                curr_close > ema_34 or 
                chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0