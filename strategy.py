#!/usr/bin/env python3
"""
6h Elder Ray + 1d Williams %R Regime Filter
Hypothesis: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
Williams %R on 1d identifies overbought/oversold conditions. Long when Bull Power > 0,
Williams %R < -80 (oversold), and price above EMA50. Short when Bear Power < 0,
Williams %R > -20 (overbought), and price below EMA50. Works in bull via long signals
on pullbacks, bear via short signals on rallies. Volume confirmation reduces false breaks.
Target: 12-37 trades/year on 6h.
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
    
    # Calculate EMA13 for Elder Ray
    if len(close) >= 13:
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema13 = np.full(n, np.nan)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Get 1d data for Williams %R regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r)
    
    # Align Williams %R with 1-bar delay (wait for 1d bar close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate EMA50 for trend filter
    if len(close) >= 50:
        ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema50 = np.full(n, np.nan)
    
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
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(ema50[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        bp = bull_power[i]
        br = bear_power[i]
        wr = williams_r_aligned[i]
        ema50_val = ema50[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_confirm = curr_volume > 1.5 * vol_ma_20
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure), Williams %R < -80 (oversold), price > EMA50, volume confirm
            long_condition = (bp > 0) and (wr < -80) and (curr_close > ema50_val) and volume_confirm
            # Short: Bear Power < 0 (selling pressure), Williams %R > -20 (overbought), price < EMA50, volume confirm
            short_condition = (br < 0) and (wr > -20) and (curr_close < ema50_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or Bear Power turns negative
            if curr_close <= entry_price - 2.5 * atr_val or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or Bull Power turns positive
            if curr_close >= entry_price + 2.5 * atr_val or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WilliamsR_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0