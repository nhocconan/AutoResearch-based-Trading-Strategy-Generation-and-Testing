#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_adx_cci_mean_reversion_v1
# Uses daily CCI to identify overbought/oversold conditions and 6h ADX to filter for weak trends (range markets).
# In range markets (ADX < 20), fades CPI extremes: longs when CCI < -100, shorts when CCI > +100.
# In trending markets (ADX >= 20), follows momentum: longs when CCI crosses above -100, shorts when CCI crosses below +100.
# This dual-regime approach aims to work in both bull (trending) and bear (range-bound) markets.
# Target: 20-60 trades/year to minimize fee drag.

name = "6h_1d_adx_cci_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI on daily timeframe
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    tp_mean = typical_price.rolling(window=20, min_periods=20).mean()
    tp_std = typical_price.rolling(window=20, min_periods=20).std()
    cci = (typical_price - tp_mean) / (0.015 * tp_std)
    cci_values = cci.values
    
    # Align CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    
    # Calculate ADX on 6h timeframe for regime detection
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        if np.isnan(cci_aligned[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        cci_val = cci_aligned[i]
        adx_val = adx[i]
        
        # Regime-based logic
        if adx_val < 20:  # Range market: mean reversion
            # Long when CCI < -100 (oversold)
            if cci_val < -100 and position != 1:
                position = 1
                signals[i] = 0.25
            # Short when CCI > +100 (overbought)
            elif cci_val > 100 and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit when CCI returns to neutral zone (-100 to 100)
            elif -100 <= cci_val <= 100 and position != 0:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:  # Trending market: momentum
            # Long when CCI crosses above -100 from below
            if cci_val > -100 and position != 1:
                # Need previous CCI to detect crossover
                if i > 50 and not np.isnan(cci_aligned[i-1]) and cci_aligned[i-1] <= -100:
                    position = 1
                    signals[i] = 0.25
            # Short when CCI crosses below +100 from above
            elif cci_val < 100 and position != -1:
                if i > 50 and not np.isnan(cci_aligned[i-1]) and cci_aligned[i-1] >= 100:
                    position = -1
                    signals[i] = -0.25
            # Exit on opposite crossover
            elif (cci_val < -100 and position == 1) or (cci_val > 100 and position == -1):
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals