# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 12-hour breakouts from daily Camarilla R3/S3 levels with 1-day EMA34 trend filter and volume confirmation.
# Targets 12h timeframe to reduce trade frequency (target: 12-37 trades/year) while using proven Camarilla structure.
# Only takes long when price breaks above R3 with volume spike and 1-day uptrend, short when breaks below S3 with volume spike and 1-day downtrend.
# Camarilla levels provide strong support/resistance, EMA34 filters trend direction, volume confirms breakout strength.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Volume spike: >2.0x 30-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    # R3 = close + 1.1*(high-low)/6
    # S3 = close - 1.1*(high-low)/6
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Align Camarilla levels to 12h timeframe (wait for daily bar to close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # 1d data for trend filter (EMA34)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above 1d EMA34
            if (close[i] > camarilla_R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below 1d EMA34
            elif (close[i] < camarilla_S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S3 and R3 OR closes below 1d EMA34
            if (close[i] > camarilla_S3_aligned[i] and close[i] < camarilla_R3_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S3 and R3 OR closes above 1d EMA34
            if (close[i] > camarilla_S3_aligned[i] and close[i] < camarilla_R3_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals