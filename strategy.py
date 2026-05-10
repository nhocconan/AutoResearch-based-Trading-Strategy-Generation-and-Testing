#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
# Hypothesis: Camarilla pivot levels (R1/S1) on 4h chart with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 with 1d uptrend and volume spike; short when breaks below S1 with 1d downtrend and volume spike.
# Uses daily trend for multi-timeframe alignment and volume confirmation to reduce false signals.
# Designed for moderate trade frequency (target: 25-50 trades/year) with clear structure.

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's close, high, low
    prev_day_close = df_1d['close'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1 for each day
    camarilla_r1 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 12
    camarilla_s1 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 12
    
    # Align to 4h timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1 for previous day, 34 for EMA, 20 for volume
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to Camarilla levels
        price_above_r1 = close[i] > camarilla_r1_aligned[i]
        price_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        if position == 0:
            # Long entry: price above R1 + daily uptrend + volume spike
            if price_above_r1 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below S1 + daily downtrend + volume spike
            elif price_below_s1 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or daily trend turns down
            if close[i] < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 or daily trend turns up
            if close[i] > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals