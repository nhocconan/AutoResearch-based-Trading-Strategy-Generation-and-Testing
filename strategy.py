#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray calculates bull power (high - EMA13) and bear power (low - EMA13)
# Long when bull power > 0 and rising, short when bear power < 0 and falling
# Uses 1d EMA50 for trend filter to avoid counter-trend trades
# Volume spike confirms institutional participation
# Works in bull/bear via trend filter - only trades in direction of higher timeframe trend
# Target: 15-30 trades/year (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Calculate 50-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 6-hour timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power > 0 and rising + volume spike + uptrend (close > EMA50_1d)
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and
                vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power < 0 and falling + volume spike + downtrend (close < EMA50_1d)
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and
                  vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder power crosses zero or trend changes
            if position == 1:
                if (bull_power_aligned[i] <= 0 or 
                    close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (bear_power_aligned[i] >= 0 or 
                    close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0