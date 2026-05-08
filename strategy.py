#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h CCI(20) mean reversion with 4h ADX filter and volume confirmation
# Uses CCI for overbought/oversold conditions, filtered by 4h ADX (>25) for trending markets
# and volume spikes to confirm reversals. Designed for low-frequency trades (15-30/year)
# to work in both bull and bear markets by fading extremes in trending conditions.

name = "1h_CCI20_4hADX25_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate CCI(20) on 1h data
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Get 4h data for ADX filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h data
    plus_dm = np.zeros_like(high_4h)
    minus_dm = np.zeros_like(low_4h)
    tr = np.zeros_like(high_4h)
    
    for i in range(1, len(high_4h)):
        high_diff = high_4h[i] - high_4h[i-1]
        low_diff = low_4h[i-1] - low_4h[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
    
    # Wilder's smoothing for ADX
    def wilder_smooth(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return result
        result[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_di14 = 100 * wilder_smooth(plus_dm, 14) / tr14
    minus_di14 = 100 * wilder_smooth(minus_dm, 14) / tr14
    dx = np.where((plus_di14 + minus_di14) > 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(cci[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: CCI < -100 (oversold) with 4h trending (ADX>25) and volume spike
            if (cci[i] < -100 and adx_aligned[i] > 25 and vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: CCI > 100 (overbought) with 4h trending (ADX>25) and volume spike
            elif (cci[i] > 100 and adx_aligned[i] > 25 and vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: CCI > -100 (exit oversold) or trend weakens
            if (cci[i] > -100 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: CCI < 100 (exit overbought) or trend weakens
            if (cci[i] < 100 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals