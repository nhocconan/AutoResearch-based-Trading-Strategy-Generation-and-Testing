#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla R1/S1 levels provide high-probability reversal zones.
Breakouts above R1 (with 1d trend confirmation) or below S1 (with 1d trend confirmation)
are traded with volume spike confirmation. Designed for 12-37 trades/year to minimize
fee drag while capturing reversals in both bull and bear markets. Works on BTC/ETH/SOL.
"""

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate 1d volume average for spike detection
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Camarilla levels from previous day's range (using 1d data)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are close, high, low of previous day
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (need extra delay as levels are based on previous day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup for EMA34 and Camarilla
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_avg_val = vol_avg_20_1d_aligned[i]
        vol_1d_val = volume_1d[i // 288] if i // 288 < len(volume_1d) else 0  # Approximate 1d volume index

        if np.isnan(camarilla_r1_val) or np.isnan(camarilla_s1_val) or np.isnan(ema_34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 with 1d uptrend (price > EMA34) and volume spike
            if close[i] > camarilla_r1_val and close[i] > ema_34_val and vol_1d_val > vol_avg_val * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with 1d downtrend (price < EMA34) and volume spike
            elif close[i] < camarilla_s1_val and close[i] < ema_34_val and vol_1d_val > vol_avg_val * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (reversal signal)
            if close[i] < camarilla_s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (reversal signal)
            if close[i] > camarilla_r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals