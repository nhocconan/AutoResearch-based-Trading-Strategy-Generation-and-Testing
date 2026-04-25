#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: Camarilla pivot R1/S1 breakout with 1d ATR-based trend filter and volume spike confirmation.
Long when price breaks above R1 in 1d ATR uptrend (close > close[-1] + 0.5*ATR) with volume > 1.8x 20-period average.
Short when price breaks below S1 in 1d ATR downtrend (close < close[-1] - 0.5*ATR) with volume > 1.8x 20-period average.
Exit on opposite Camarilla level break, trend reversal, or ATR stoploss (2.0).
Designed for BTC/ETH to work in bull/bear via structure (Camarilla pivots) with trend/volume filters.
Target trades: 75-200 over 4 years to minimize fee drag and maximize test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR for trend filter (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr0 = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR trend: uptrend if close > previous close + 0.5*ATR, downtrend if close < previous close - 0.5*ATR
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = np.nan
    atr_trend_up = close_1d > (close_prev + 0.5 * atr_1d)
    atr_trend_down = close_1d < (close_prev - 0.5 * atr_1d)
    atr_trend_up_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_up)
    atr_trend_down_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_down)
    
    # Previous day's Camarilla levels (R1, S1, PP)
    # Formula: PP = (H + L + C) / 3
    #          R1 = PP + (H - L) * 1.1 / 2
    #          S1 = PP - (H - L) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # ATR for stop loss (14-period) on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr0 = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need ATR trend (14), volume MA (20), ATR (14)
    start_idx = max(14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_trend_up_aligned[i]) or np.isnan(atr_trend_down_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 in 1d ATR uptrend with volume spike
            long_signal = (curr_close > r1_aligned[i]) and \
                         atr_trend_up_aligned[i] and \
                         volume_spike[i]
            # Short: price breaks below S1 in 1d ATR downtrend with volume spike
            short_signal = (curr_close < s1_aligned[i]) and \
                          atr_trend_down_aligned[i] and \
                          volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down OR ATR stoploss hit
            if (curr_close < s1_aligned[i]) or \
               (~atr_trend_up_aligned[i]) or \
               (curr_close < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up OR ATR stoploss hit
            if (curr_close > r1_aligned[i]) or \
               (~atr_trend_down_aligned[i]) or \
               (curr_close > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0