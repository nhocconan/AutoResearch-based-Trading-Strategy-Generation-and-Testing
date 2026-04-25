#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d EMA50 trend + volume spike + chop filter
Hypothesis: Donchian breakouts capture strong momentum; 1d EMA50 filters trend direction;
volume spike confirms institutional participation; chop filter avoids whipsaws in ranging markets.
Works in bull via long breakouts in uptrend, bear via short breakouts in downtrend.
Target: 20-50 trades/year on 4h timeframe.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate Bollinger Band Width for chop regime (20, 2)
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        bb_width = (upper_band - lower_band) / sma_20
        # Chop regime: bb_width > 50th percentile of last 100 bars = ranging market
        bb_width_percentile = np.zeros_like(bb_width)
        for i in range(20, n):
            window = bb_width[max(0, i-99):i+1]
            if len(window) > 0:
                bb_width_percentile[i] = (np.sum(window <= bb_width[i]) / len(window)) * 100
        chop_condition = bb_width_percentile > 50  # True when choppy/ranging
    else:
        chop_condition = np.zeros(n, dtype=bool)
        bb_width_percentile = np.zeros(n)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(sma_20[i]) if i >= 20 else False):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_aligned[i]
        atr_val = atr[i]
        is_chop = chop_condition[i] if i < len(chop_condition) else False
        
        # Donchian channels (20-period)
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1]) if i >= 0 else 0
            donchian_low = np.min(low[:i+1]) if i >= 0 else 0
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1]) if i >= 0 else 0
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND uptrend AND not choppy
            long_condition = (curr_high > donchian_high) and volume_spike and (curr_close > ema_50) and (not is_chop)
            # Short: price breaks below Donchian low AND volume spike AND downtrend AND not choppy
            short_condition = (curr_low < donchian_low) and volume_spike and (curr_close < ema_50) and (not is_chop)
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit long: stoploss (2.5*ATR below highest) or trend reversal or chop regime
            if (curr_close <= highest_since_entry - 2.5 * atr_val) or (curr_close < ema_50) or is_chop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit short: stoploss (2.5*ATR above lowest) or trend reversal or chop regime
            if (curr_close >= lowest_since_entry + 2.5 * atr_val) or (curr_close > ema_50) or is_chop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_EMA50_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0