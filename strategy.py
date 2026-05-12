# 4h_12h_Camarilla_R3S3_Breakout_TrendVol_v1
# Hypothesis: Camarilla pivot breakouts with 12h trend filter and volume spikes. Works in bull/bear via trend filter and volume confirmation.
# Targets 4h timeframe with 12h trend filter to reduce whipsaw. Camarilla R3/S3 levels provide strong support/resistance.
# Volume spikes confirm breakout strength. Designed for 20-40 trades/year to avoid fee drag.

name = "4h_12h_Camarilla_R3S3_Breakout_TrendVol_v1"
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
    
    # Volume spike: >2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's close, high, low for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3, S3
    # R3 = Close + 1.1*(High-Low)*1.1/2
    # S3 = Close - 1.1*(High-Low)*1.1/2
    camarilla_range = prev_high - prev_low
    camarilla_R3 = prev_close + camarilla_range * 1.1 * 1.1 / 2
    camarilla_S3 = prev_close - camarilla_range * 1.1 * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(prev_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above 12h EMA50
            if (close[i] > camarilla_R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below 12h EMA50
            elif (close[i] < camarilla_S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S3 and R3 OR closes below 12h EMA50
            if (close[i] > camarilla_S3_aligned[i] and close[i] < camarilla_R3_aligned[i]) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S3 and R3 OR closes above 12h EMA50
            if (close[i] > camarilla_S3_aligned[i] and close[i] < camarilla_R3_aligned[i]) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals