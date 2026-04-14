#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with 1-day trend filter and volume confirmation
# Uses Elder Ray (Bull Power/Bear Power) from 6h data for entry signals
# Daily EMA(50) as trend filter (only long when price > EMA50, short when price < EMA50)
# Volume confirmation > 1.5x 20-period EMA to reduce false signals
# Designed for 15-30 trades/year with clear momentum logic
# Works in bull markets via bull power + uptrend and in bear markets via bear power + downtrend
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray components (13-period EMA for EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(13, n):
        # Get aligned daily EMA50
        ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)[i]
        
        if np.isnan(ema50_1d_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Trend filter: only long in uptrend, only short in downtrend
        uptrend = close[i] > ema50_1d_aligned
        downtrend = close[i] < ema50_1d_aligned
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Bull Power > 0 + uptrend + volume confirmation
        if position == 0 and bull_power[i] > 0 and uptrend and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Bear Power < 0 + downtrend + volume confirmation
        elif position == 0 and bear_power[i] < 0 and downtrend and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite signal or loss of momentum
        elif position != 0:
            if position == 1 and (bear_power[i] < 0 or not uptrend):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (bull_power[i] > 0 or not downtrend):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0