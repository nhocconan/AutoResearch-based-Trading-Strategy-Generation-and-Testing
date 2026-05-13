# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: At 12h timeframe, buy when price breaks above Camarilla R3 level and sell when breaks below S3 level,
# but only in the direction of the 1d trend (EMA34) and with volume confirmation (>1.5x 20-period average).
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets.
# Low frequency due to strict Camarilla breakout requirement and volume confirmation.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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

    # Get 1d data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R3, S3) for each 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    daily_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + 1.1 * daily_range / 2
    camarilla_s3 = df_1d['close'] - 1.1 * daily_range / 2
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend conditions
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above R3 + uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R3 OR trend reversal
            if close[i] < camarilla_r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 OR trend reversal
            if close[i] > camarilla_s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals