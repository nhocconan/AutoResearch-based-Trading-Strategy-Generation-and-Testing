#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
    # Works in bull/bear by trading breakouts only in expanding volatility regimes
    # Volume spike confirms institutional interest, chop filter avoids whipsaws in ranging markets
    # Target: 20-30 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility regime
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Align 1d ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d choppiness index (CHOP) for regime detection
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = np.sum(tr[i-14:i+1])
        hh = np.max(high_1d[i-14:i+1])
        ll = np.min(low_1d[i-14:i+1])
        if hh != ll and atr_sum > 0:
            chop_1d[i] = 100 * np.log10(atr_sum / np.log10(14) / (hh - ll))
        else:
            chop_1d[i] = 50.0  # neutral
    
    # Align 1d CHOP to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 1d volume spike detector (current volume > 2.0 * 20-period average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_spike_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        if not np.isnan(vol_ma_20_1d[i]) and vol_ma_20_1d[i] > 0:
            vol_spike_1d[i] = volume_1d[i] / vol_ma_20_1d[i]
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Previous day's Camarilla pivot levels
    camarilla_h4 = np.full(len(df_1d), np.nan)  # resistance
    camarilla_l4 = np.full(len(df_1d), np.nan)  # support
    for i in range(1, len(df_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            camarilla_h4[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
            camarilla_l4[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when CHOP > 50 (trending market) 
        regime_filter = chop_1d_aligned[i] > 50
        
        # Volume confirmation: volume spike > 1.5
        volume_confirm = vol_spike_1d_aligned[i] > 1.5
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h4_aligned[i]
        breakout_short = close[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        long_entry = breakout_long and regime_filter and volume_confirm
        short_entry = breakout_short and regime_filter and volume_confirm
        
        # Exit conditions: opposite breakout or regime change to choppy
        long_exit = (close[i] < camarilla_l4_aligned[i]) or (chop_1d_aligned[i] < 40)
        short_exit = (close[i] > camarilla_h4_aligned[i]) or (chop_1d_aligned[i] < 40)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0