#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Trend Filter + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing, 1d EMA50 uptrend, volume > 1.5x 20-bar average
# Short when Bear Power < 0 and decreasing, 1d EMA50 downtrend, volume > 1.5x 20-bar average
# Exit when power crosses zero or reverses
# Uses discrete sizing (0.25) to control drawdown and fee churn
# Works in bull/bear by requiring 1d trend alignment
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_ElderRay_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient history for volume MA and EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter
        ema_trend_up = close[i] > ema_50_1d_aligned[i]
        ema_trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray momentum
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising, 1d EMA50 uptrend, volume confirm
            if bull_power[i] > 0 and bull_rising and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 and falling, 1d EMA50 downtrend, volume confirm
            elif bear_power[i] < 0 and bear_falling and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Bull Power <= 0 or stops rising
            if bull_power[i] <= 0 or not bull_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Bear Power >= 0 or stops falling
            if bear_power[i] >= 0 or not bear_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals