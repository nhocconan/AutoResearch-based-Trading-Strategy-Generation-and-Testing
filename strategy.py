#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot levels (R1/S1) on 1d act as strong support/resistance. 
Breakout above R1 or below S1 with volume confirmation and aligned 1d trend (close > EMA34) signals continuation. 
Designed for low trade frequency (~20-40/year) to minimize fee drag in 4-hour bars.
Works in both bull and bear markets by following trend + momentum.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = H - L
    range_ = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    # R1 = Close + (Range * 1.1/12)
    # S1 = Close - (Range * 1.1/12)
    camarilla_r1 = df_1d['close'] + (range_ * 1.1 / 12)
    camarilla_s1 = df_1d['close'] - (range_ * 1.1 / 12)
    
    # Align to 4h - use previous day's levels (available at 4h open)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        if position == 0:
            # LONG: Price breaks above R1, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 (failed breakout) OR volume drops
            if (close[i] < camarilla_r1_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 (failed breakdown) OR volume drops
            if (close[i] > camarilla_s1_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals