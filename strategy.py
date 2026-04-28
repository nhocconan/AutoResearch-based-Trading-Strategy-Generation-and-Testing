#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Trend Filter + Volume Spike
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising, price > 1d EMA50 (uptrend), volume > 2.0x 20-bar average
# Short when Bear Power < 0 and falling, price < 1d EMA50 (downtrend), volume confirmation
# Exit when power crosses zero (Bull Power <= 0 for long, Bear Power >= 0 for short)
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures buying/selling pressure relative to trend. Works in both bull/bear markets
# by requiring alignment with higher-timeframe trend (1d EMA50). Volume confirmation filters weak signals.
# Zero-cross exit provides timely reversal signals.

name = "6h_ElderRay_BullBearPower_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_1d_aligned[i]
        ema_trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray momentum: rising/falling power
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising, price > 1d EMA50 (uptrend), volume confirm
            if bull_power[i] > 0 and bull_rising and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 and falling, price < 1d EMA50 (downtrend), volume confirm
            elif bear_power[i] < 0 and bear_falling and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit when Bull Power <= 0
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit when Bear Power >= 0
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals