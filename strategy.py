#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band squeeze breakout with daily volume confirmation and ADX trend filter.
Long when price breaks above upper BB after low volatility squeeze (BBWidth < 20th percentile) and ADX > 25.
Short when price breaks below lower BB after squeeze with ADX > 25.
Exit when price crosses middle BB (20 SMA) or volatility expands (BBWidth > 80th percentile).
Uses daily volume filter to ensure institutional participation. Works in both bull and bear markets
by trading volatility breakouts in trending regimes (ADX > 25) while avoiding chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bb_width = (upper_band - lower_band) / sma
    
    # Percentile ranks for BB width (20 and 80)
    bb_width_series = pd.Series(bb_width)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    bb_width_80th = bb_width_series.rolling(window=50, min_periods=20).quantile(0.80).values
    
    # ADX (14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_sm = np.zeros_like(plus_dm)
        minus_dm_sm = np.zeros_like(minus_dm)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sm[period-1] = np.mean(plus_dm[:period])
        minus_dm_sm[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sm[i] = (plus_dm_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_sm[i] = (minus_dm_sm[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_sm / (atr + 1e-10)
        minus_di = 100 * minus_dm_sm / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(sma[i]) or np.isnan(bb_width[i]) or np.isnan(adx[i]) or np.isnan(bb_width_20th[i]) or np.isnan(bb_width_80th[i]) or np.isnan(avg_vol_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for Bollinger Band squeeze breakout with volume and trend confirmation
            squeeze_condition = bb_width[i] < bb_width_20th[i]
            volume_condition = volume_1d[i] > avg_vol_1d_aligned[i]
            trend_condition = adx[i] > 25
            
            if squeeze_condition and volume_condition and trend_condition:
                if close[i] > upper_band[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower_band[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle Bollinger Band or volatility expands
                if close[i] < sma[i] or bb_width[i] > bb_width_80th[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle Bollinger Band or volatility expands
                if close[i] > sma[i] or bb_width[i] > bb_width_80th[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BB_Squeeze_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0