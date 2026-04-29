#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Trend + Volume Spike
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure bull/bear strength
# Weekly EMA50 trend filter ensures alignment with primary trend (avoids counter-trend trades)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete position sizing (0.25) minimizes fee churn
# Designed for low trade frequency (target: 12-37 trades/year) on 6h timeframe
# Works in bull markets: Long when Bull Power > 0, price > weekly EMA50, volume spike
# Works in bear markets: Short when Bear Power < 0, price < weekly EMA50, volume spike
# Elder Ray excels in volatile markets by measuring power behind moves

name = "6h_ElderRay_WeeklyEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = np.zeros(n)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Exit conditions: reverse signal or loss of momentum
        if position == 1:  # Long position
            # Exit if Bull Power turns negative OR weekly trend turns bearish
            if bull_power[i] <= 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if Bear Power turns positive OR weekly trend turns bullish
            if bear_power[i] >= 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND price > weekly EMA50 AND volume spike
            if bull_power[i] > 0 and close[i] > ema_50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND price < weekly EMA50 AND volume spike
            elif bear_power[i] < 0 and close[i] < ema_50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals