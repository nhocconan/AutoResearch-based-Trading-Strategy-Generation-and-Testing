#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (1.8x)
# Long when price breaks above 1d Camarilla R1 AND price > 1d EMA34 (uptrend) AND volume > 1.8x 20-period average
# Short when price breaks below 1d Camarilla S1 AND price < 1d EMA34 (downtrend) AND volume > 1.8x 20-period average
# Exit when price crosses 1d Camarilla pivot point OR 1d EMA34 filter reverses
# Uses Camarilla pivot levels from daily timeframe for structure + volume confirmation to reduce false breakouts
# 1d EMA34 provides strong trend filter for BTC/ETH in both bull and bear markets
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Timeframe: 4h (primary)

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_1.8x"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for volume calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    volume_4h = df_4h['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12, PP = (High+Low+Close)/3
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 4h (threshold: 1.8x for balanced frequency)
    if len(volume_4h) >= 20:
        vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume_4h > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(prices), dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND price > EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND price < EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot OR price < EMA34 (trend weakening)
            if close[i] < camarilla_pp_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot OR price > EMA34 (trend weakening)
            if close[i] > camarilla_pp_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals