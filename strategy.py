#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ATR Filter + Volume Spike + Chop Filter
Hypothesis: Donchian channel breakouts capture sustained momentum, filtered by 1d ATR regime (only trade when volatility is elevated), volume confirmation ensures institutional participation, and chop filter avoids ranging markets. This combination works in both bull and bear markets by adapting to volatility regimes while maintaining strict entry criteria to limit trades to optimal range (20-50/year).
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
    
    # Get 1d data for ATR filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    tr1 = pd.Series(daily_high).diff().abs()
    tr2 = (pd.Series(daily_high) - pd.Series(daily_close).shift()).abs()
    tr3 = (pd.Series(daily_low) - pd.Series(daily_close).shift()).abs()
    tr_daily = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr_daily.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 20-period ATR for stoploss (4h timeframe)
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_4h = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr_4h = np.full(n, 0.0)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (CHOP) for regime filter
    if len(close) >= 14:
        atr_sum = pd.Series(atr_4h).rolling(window=14, min_periods=14).sum()
        hh = pd.Series(high).rolling(window=14, min_periods=14).max()
        ll = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        chop_values = chop.values
    else:
        chop_values = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, ATR, and volume MA to propagate
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_4h[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_1d = atr_14_1d_aligned[i]
        atr_4h_val = atr_4h[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma = vol_ma_20[i]
        chop = chop_values[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        # Volatility filter: only trade when 1d ATR is above its 50-period MA (elevated volatility)
        if i >= 50:
            atr_ma_50 = np.mean(atr_14_1d_aligned[max(0, i-49):i+1])
            vol_filter = atr_1d > atr_ma_50
        else:
            vol_filter = True  # Not enough data for MA, allow trade
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volume spike AND volatility filter AND trending regime
            long_condition = (curr_close > upper) and volume_spike and vol_filter and trending_regime
            # Short: price breaks below Donchian lower AND volume spike AND volatility filter AND trending regime
            short_condition = (curr_close < lower) and volume_spike and vol_filter and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below Donchian lower (reversal signal)
            if curr_close <= entry_price - 2.0 * atr_4h_val or curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above Donchian upper (reversal signal)
            if curr_close >= entry_price + 2.0 * atr_4h_val or curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Filter_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0