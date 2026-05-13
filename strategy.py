#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS
# Hypothesis: Trade Camarilla pivot breakouts with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R1 with volume spike and 1d EMA34 uptrend.
# Short when price breaks below S1 with volume spike and 1d EMA34 downtrend.
# Uses Camarilla levels from daily timeframe for structure, volume spike for confirmation,
# and EMA34 trend filter to avoid counter-trend trades. Designed for 4h timeframe to
# balance trade frequency and signal quality in both bull and bear markets.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS"
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

    # Get 1d data for Camarilla calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are close, high, low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike: current volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma

    # Align all 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Price action
        price_high = high[i]
        price_low = low[i]
        price_close = close[i]

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and EMA34 uptrend
            if (price_high > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                price_close > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and EMA34 downtrend
            elif (price_low < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  price_close < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or loses volume/spike or trend
            if (price_low < camarilla_s1_aligned[i] or 
                not volume_spike[i] or 
                price_close < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or loses volume/spike or trend
            if (price_high > camarilla_r1_aligned[i] or 
                not volume_spike[i] or 
                price_close > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals