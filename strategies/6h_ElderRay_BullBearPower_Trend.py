# 6h_ElderRay_BullBearPower_Trend
# Hypothesis: Elder Ray (Bull/Bear Power) with 60-day EMA trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 and rising, Bear Power < 0 and falling in uptrend.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising in downtrend.
# Uses 1d EMA60 for trend filter and 1d volume spike for confirmation to reduce whipsaws.
# Works in bull markets by buying dips with bullish momentum, and in bear markets by selling rallies with bearish momentum.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_ElderRay_BullBearPower_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray components: Bull Power and Bear Power using EMA13
    period_ema13 = 13
    ema13 = np.full(n, np.nan)
    if n >= period_ema13:
        ema13[period_ema13-1] = np.mean(close[:period_ema13])
        alpha = 2 / (period_ema13 + 1)
        for i in range(period_ema13, n):
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # 1d EMA60 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema60_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 60:
        ema60_1d[59] = np.mean(close_1d[:60])
        alpha = 2 / (60 + 1)
        for i in range(60, len(close_1d)):
            ema60_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema60_1d[i-1]
    ema60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema60_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_ema13, 60)  # Ensure EMA13 and 1d indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(ema60_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 4.0  # 4x 6h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend determination: price vs 1d EMA60
        is_uptrend = close[i] > ema60_1d_aligned[i]
        is_downtrend = close[i] < ema60_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, in uptrend with volume
            if (bull_power[i] > 0 and 
                i > start_idx and bull_power[i] > bull_power[i-1] and
                bear_power[i] < 0 and
                is_uptrend and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0, in downtrend with volume
            elif (bear_power[i] > 0 and  # Bear Power is positive when EMA13 > Low (bearish)
                  i > start_idx and bear_power[i] > bear_power[i-1] and  # Rising bear power = more bearish
                  bull_power[i] > 0 and
                  is_downtrend and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or trend changes
            if (bull_power[i] <= 0 or 
                not is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns negative or trend changes
            if (bear_power[i] <= 0 or 
                not is_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals