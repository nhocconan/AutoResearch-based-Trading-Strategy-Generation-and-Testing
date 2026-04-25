#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels from prior day act as key support/resistance.
Breakout above H3 with 1d EMA34 uptrend, volume spike, and choppy regime (CHOP > 61.8)
captures bullish momentum in range-bound markets. Breakdown below L3 with 1d EMA34
downtrend, volume spike, and choppy regime captures bearish momentum. Chop filter
avoids trending markets where false breakouts occur. Works in bull via long
breakouts, bear via short breakdowns. Volume spike confirms participation.
Target: 20-50 trades/year on 4h.
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
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # H3 = Close + 1.1*(High-Low)/4
    # L3 = Close - 1.1*(High-Low)/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels with 1-bar delay (wait for 1d bar close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
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
    
    # Calculate Choppiness Index (CHOP) on 4h data
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_sum = tr.rolling(window=14, min_periods=14).sum()
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
        chop_values = chop.fillna(50).values  # fill NaN with 50 (neutral)
    else:
        chop_values = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        atr_val = atr[i]
        chop_val = chop_values[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Chop filter: CHOP > 61.8 indicates ranging market (good for breakout fade/mean reversion)
        # In ranging markets, breakouts often fail quickly, so we use CHOP > 61.8 to confirm
        # we are in a range where Camarilla levels are more likely to hold
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Long: break above H3 AND uptrend AND volume spike AND choppy regime
            long_condition = curr_high > h3 and curr_close > ema_34 and volume_spike and chop_filter
            # Short: break below L3 AND downtrend AND volume spike AND choppy regime
            short_condition = curr_low < l3 and curr_close < ema_34 and volume_spike and chop_filter
            
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

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0