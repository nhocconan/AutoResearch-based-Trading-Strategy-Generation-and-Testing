#!/usr/bin/env python3
"""
12h Camarilla R1S1 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance. Breakouts with 
volume confirmation and 1d EMA34 trend filter capture momentum moves. Choppiness 
index regime filter avoids whipsaws in ranging markets. Designed for 12h timeframe 
to target 50-150 total trades over 4 years (12-37/year) with discrete position 
sizing (0.25) to control drawdown in both bull and bear markets.
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
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
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
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Calculate Choppiness Index (14) for regime filter
    chop = np.full(n, 50.0)  # default to neutral
    if len(close) >= 14:
        atr_sum = np.zeros(n)
        for i in range(n):
            if i >= 13:
                atr_sum[i] = np.sum(atr[i-13:i+1])
            else:
                atr_sum[i] = np.sum(atr[0:i+1]) if i > 0 else atr[0]
        
        highest_high = np.zeros(n)
        lowest_low = np.zeros(n)
        for i in range(n):
            if i >= 13:
                highest_high[i] = np.max(high[i-13:i+1])
                lowest_low[i] = np.min(low[i-13:i+1])
            else:
                highest_high[i] = np.max(high[0:i+1]) if i >= 0 else high[0]
                lowest_low[i] = np.min(low[0:i+1]) if i >= 0 else low[0]
        
        range_14 = highest_high - lowest_low
        # Avoid division by zero
        chop = np.where(
            (range_14 > 0) & (atr_sum > 0),
            100 * np.log10(atr_sum / range_14) / np.log10(14),
            50.0
        )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Camarilla calculation (uses prior day) and indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop[i])):
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
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        
        # Calculate Camarilla levels for R1 and S1 using prior 12h bar's range
        # Camarilla: based on previous period's high-low-close
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla R1 and S1 levels
            r1 = prev_close + range_val * 1.1 / 12
            s1 = prev_close - range_val * 1.1 / 12
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        # For breakout strategy, we want trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Long: break above Camarilla R1 AND uptrend AND volume spike AND trending regime
            long_condition = (curr_close > r1) and (curr_close > ema_34) and volume_spike and trending_regime
            # Short: break below Camarilla S1 AND downtrend AND volume spike AND trending regime
            short_condition = (curr_close < s1) and (curr_close < ema_34) and volume_spike and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below Camarilla S1 or EMA34
            if (curr_close <= entry_price - 2.0 * atr_val) or (curr_close < s1) or (curr_close < ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above Camarilla R1 or EMA34
            if (curr_close >= entry_price + 2.0 * atr_val) or (curr_close > r1) or (curr_close > ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0