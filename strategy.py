#!/usr/bin/env python3
"""
4h_1d_Pullback_To_EMA21_with_Volume_Spice
Hypothesis: Buy pullbacks to EMA21 in uptrends (EMA21 > EMA50) and sell pullbacks in downtrends (EMA21 < EMA50). Use volume spike (1.5x 20-period average) for entry confirmation and 1-day ADX > 25 to ensure trending market. Exit when price crosses EMA8 or ADX drops below 20. Designed for 20-35 trades/year with clear trend-following logic that works in bull (continuations) and bear (mean reversion within trend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pullback_To_EMA21_with_Volume_Spice"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR ADX TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4H INDICATORS ===
    # EMA21, EMA50, EMA8
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema8 = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(ema8[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_aligned[i] > 25
        
        # Volume spike
        volume_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Uptrend: EMA21 > EMA50
        uptrend = ema21[i] > ema50[i]
        # Downtrend: EMA21 < EMA50
        downtrend = ema21[i] < ema50[i]
        
        # Long: pullback to EMA21 in uptrend with volume spike
        long_signal = (close[i] <= ema21[i] * 1.005 and  # within 0.5% above EMA21
                      close[i] >= ema21[i] * 0.995 and   # within 0.5% below EMA21
                      uptrend and 
                      volume_spike and 
                      trending)
        
        # Short: pullback to EMA21 in downtrend with volume spike
        short_signal = (close[i] <= ema21[i] * 1.005 and 
                       close[i] >= ema21[i] * 0.995 and
                       downtrend and 
                       volume_spike and 
                       trending)
        
        # Exit: price crosses EMA8 or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < ema8[i] or adx_aligned[i] < 20))
        exit_short = (position == -1 and 
                     (close[i] > ema8[i] or adx_aligned[i] < 20))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals