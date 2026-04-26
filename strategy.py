#!/usr/bin/env python3
"""
6h_ADX_DMI_VolumeSpike_1dTrend
Hypothesis: Use ADX(14) > 25 for trending regime on 6h timeframe, with +DI > -DI for long and -DI > +DI for short. 
Add 1d EMA50 trend filter to ensure alignment with higher timeframe trend. 
Require volume > 2.0x 20-period MA for entry to avoid false breakouts. 
Exit when ADX < 20 (trend weakening) or 1d trend changes. 
Designed to capture strong trends in both bull and bear markets while avoiding choppy regimes.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # 6h ADX calculation
    # +DM, -DM, TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Wilder's smoothing: result[i] = (result[i-1] * (period-1) + data[i]) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    smoothed_plus_dm = wilders_smoothing(plus_dm, period_adx)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period_adx)
    smoothed_tr = wilders_smoothing(tr, period_adx)
    
    # Avoid division by zero
    plus_di = np.where(smoothed_tr != 0, (smoothed_plus_dm / smoothed_tr) * 100, 0)
    minus_di = np.where(smoothed_tr != 0, (smoothed_minus_dm / smoothed_tr) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)
    adx = wilders_smoothing(dx, period_adx)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA + 14*2 for ADX + 20 for volume MA)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend) AND +DI > -DI (bullish) with 1d uptrend and volume spike
            if (adx[i] > 25 and plus_di[i] > minus_di[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) AND -DI > +DI (bearish) with 1d downtrend and volume spike
            elif (adx[i] > 25 and minus_di[i] > plus_di[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX < 20 (trend weakening) OR 1d trend changes to downtrend OR -DI > +DI
            if (adx[i] < 20 or not uptrend_1d[i] or minus_di[i] > plus_di[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX < 20 (trend weakening) OR 1d trend changes to uptrend OR +DI > -DI
            if (adx[i] < 20 or not downtrend_1d[i] or plus_di[i] > minus_di[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0