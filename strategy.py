#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R1 AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S1 AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price retracement to Camarilla pivot point OR EMA34(1d) trend flip
# Uses 4h primary timeframe with 1d HTF for trend filter to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Camarilla levels provide intraday support/resistance; volume confirmation filters false breaks; 1d EMA avoids counter-trend trades

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # For each 4h bar, use the prior completed 1d bar's OHLC
    camarilla_pivot = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    if len(df_1d) >= 1:
        # Get 1d OHLC arrays
        o_1d = df_1d['open'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # Calculate Camarilla levels for each 1d bar
        camarilla_pivot_1d = (h_1d + l_1d + c_1d) / 3.0
        camarilla_r1_1d = camarilla_pivot_1d + 1.1 * (h_1d - l_1d) / 12.0
        camarilla_s1_1d = camarilla_pivot_1d - 1.1 * (h_1d - l_1d) / 12.0
        
        # Align to 4h timeframe (each 1d bar = 6 four-hour bars)
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
        camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
        camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
        
        camarilla_pivot = camarilla_pivot_aligned
        camarilla_r1 = camarilla_r1_aligned
        camarilla_s1 = camarilla_s1_aligned
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND close > EMA34(1d) AND volume spike
            if (high[i] > camarilla_r1[i] and  # breakout above R1 level
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND close < EMA34(1d) AND volume spike
            elif (low[i] < camarilla_s1[i] and  # breakout below S1 level
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Camarilla pivot OR close < EMA34(1d) (trend flip)
            if close[i] <= camarilla_pivot[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Camarilla pivot OR close > EMA34(1d) (trend flip)
            if close[i] >= camarilla_pivot[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals