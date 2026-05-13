#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_Filter
Hypothesis: Camarilla R3/S3 levels on 4h act as strong support/resistance. A breakout above R3 or below S3 with volume confirmation and aligned 4h trend (close > EMA34) signals continuation. Uses 4h for signal direction and 1h for precise entry timing. Session filter (08-20 UTC) reduces noise. Targets 15-30 trades/year to minimize fee drag.
"""

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Filter"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivots and trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels from previous 4h bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    # Range = H - L
    range_ = df_4h['high'] - df_4h['low']
    
    # Camarilla levels
    # R3 = Close + (Range * 1.1/2)
    # S3 = Close - (Range * 1.1/2)
    camarilla_r3 = df_4h['close'] + (range_ * 1.1 / 2)
    camarilla_s3 = df_4h['close'] - (range_ * 1.1 / 2)
    
    # Align to 1h - use previous 4h bar's levels (available at 1h open after 4h close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3.values)
    
    # 4h trend filter: EMA(34) on close
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume confirmation: current volume > 1.5x 24-period average (~1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Price breaks above R3, volume confirmation, price above 4h EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3, volume confirmation, price below 4h EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 (failed breakout) OR volume drops
            if (close[i] < camarilla_r3_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 (failed breakdown) OR volume drops
            if (close[i] > camarilla_s3_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals