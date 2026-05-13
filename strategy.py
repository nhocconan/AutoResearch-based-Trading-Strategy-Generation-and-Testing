# 165079
#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
Hypothesis: Camarilla pivot levels (R3/S3) on 1d act as strong support/resistance. A breakout above R3 or below S3 with volume confirmation and aligned 1d trend (close > EMA34) signals continuation. Uses 60% position size to balance risk/return. Designed for low trade frequency (~20-40/year) to minimize fee drag in 6-8 hour bars.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
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
    # R3 = Close + (Range * 1.1/2)
    # S3 = Close - (Range * 1.1/2)
    camarilla_r3 = df_1d['close'] + (range_ * 1.1 / 2)
    camarilla_s3 = df_1d['close'] - (range_ * 1.1 / 2)
    
    # Align to 6t - use previous day's levels (available at 6t open)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
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
            # LONG: Price breaks above R3, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.60
                position = 1
            # SHORT: Price breaks below S3, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.60
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
                signals[i] = 0.60
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 (failed breakdown) OR volume drops
            if (close[i] > camarilla_s3_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.60
    
    return signals