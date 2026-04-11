#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_pivot_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 20-period CCI on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_20 = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad_20 = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_20 = (tp_1d - sma_20) / (0.015 * mad_20)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after EMA50 and CCI warmup
        # Skip if any required data is invalid
        if (np.isnan(cci_20[i]) or np.isnan(sma_20[i]) or np.isnan(mad_20[i]) or
            np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Daily pivot points (using previous day's data)
        if i > 0:
            prev_high = high_1d[i-1] if i-1 < len(high_1d) else high_1d[-1]
            prev_low = low_1d[i-1] if i-1 < len(low_1d) else low_1d[-1]
            prev_close = close_1d[i-1] if i-1 < len(close_1d) else close_1d[-1]
            pivot = (prev_high + prev_low + prev_close) / 3.0
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
        else:
            pivot = r1 = s1 = 0
        
        # Long conditions: CCI > 100 AND price > pivot R1 with volume
        long_signal = volume_confirmed and (cci_20[i] > 100) and (price_close > r1)
        
        # Short conditions: CCI < -100 AND price < pivot S1 with volume
        short_signal = volume_confirmed and (cci_20[i] < -100) and (price_close < s1)
        
        # Exit when CCI crosses zero
        exit_long = position == 1 and cci_20[i] <= 0
        exit_short = position == -1 and cci_20[i] >= 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 12h CCI(20) breakout with daily pivot levels and volume confirmation.
# Uses daily CCI to identify overbought/oversold conditions (>100/<-100).
# Enters long when CCI > 100 and price breaks above daily R1 pivot with volume confirmation.
# Enters short when CCI < -100 and price breaks below daily S1 pivot with volume confirmation.
# Exits when CCI crosses zero. Works in both bull and bear markets by capturing
# momentum extremes. Daily pivot levels provide structural support/resistance.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades
# over 4 years to minimize fee drag on 12h timeframe. CCI(20) is effective for
# identifying trend exhaustion and continuation patterns in crypto markets.