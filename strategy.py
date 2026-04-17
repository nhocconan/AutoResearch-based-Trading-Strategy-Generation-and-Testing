#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with volume confirmation and 1d ADX regime filter.
Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + (ADX < 20 range regime OR ADX > 25 with price > EMA50).
Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + (ADX < 20 range regime OR ADX > 25 with price < EMA50).
Exit when price crosses Donchian midpoint or regime shifts to opposite trend.
Uses 1d for ADX/EMA regime, 4h for Donchian/volume/EMA.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 1d data for regime filters (ADX, EMA)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h indicators
    period = 20
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donch_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        adx_val = adx_14_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        price = close[i]
        
        # Range regime: ADX < 20
        # Trend regime: ADX > 25 and price > EMA50 (for long) or price < EMA50 (for short)
        is_range = adx_val < 20
        is_trend_long = adx_val > 25 and price > ema50_1d_val
        is_trend_short = adx_val > 25 and price < ema50_1d_val
        
        # Breakout and volume conditions
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        breakout_up = close[i] > donch_high[i-1]  # break above previous period's high
        breakout_down = close[i] < donch_low[i-1]  # break below previous period's low
        
        if position == 0:
            # Long: bullish breakout + volume + (range regime OR trend regime long)
            if breakout_up and vol_ok and (is_range or is_trend_long):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume + (range regime OR trend regime short)
            elif breakout_down and vol_ok and (is_range or is_trend_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midpoint OR regime shifts to trend short
            if close[i] < donch_mid[i] or is_trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midpoint OR regime shifts to trend long
            if close[i] > donch_mid[i] or is_trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_1dADXEMA_Regime"
timeframe = "4h"
leverage = 1.0