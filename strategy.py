#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Elder Ray Index (Bull Power/Bear Power) with 1d EMA trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > 1d EMA50 AND volume > 1.5 * avg_volume(20)
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when Bull Power and Bear Power converge (both near zero) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power relative to EMA13 to detect momentum shifts
# 1d EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume confirmation ensures institutional participation
# Works in bull markets (buying on bull power in uptrend) and bear markets (selling on bear power in downtrend)

name = "6h_ElderRay_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA50 AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA50 AND volume confirmation
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Power convergence (both near zero) OR volume drops below average
            if (abs(bull_power[i]) < 0.1 * close[i] and abs(bear_power[i]) < 0.1 * close[i]) or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Power convergence OR volume drops below average
            if (abs(bull_power[i]) < 0.1 * close[i] and abs(bear_power[i]) < 0.1 * close[i]) or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals