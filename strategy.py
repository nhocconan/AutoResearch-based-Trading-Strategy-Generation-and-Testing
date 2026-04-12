#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_volume_trend_v1
# Uses daily Camarilla pivot levels (H4/L4) as support/resistance on 4h chart.
# Long when price breaks above H4 with volume confirmation (volume > 1.5x 20-period avg) AND 4h close > 20-period EMA (trend filter).
# Short when price breaks below L4 with volume confirmation AND 4h close < 20-period EMA.
# Exits when price returns to daily pivot point (PP) or when trend filter reverses.
# Designed for moderate trade frequency (target: 20-50 trades/year) to balance signal quality and cost.
# Works in trending markets via breakouts with trend confirmation and ranging markets via mean reversion to pivot.

name = "4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = PP + 1.1/2 * range, L4 = PP - 1.1/2 * range
    h4 = pp + (1.1 / 2) * range_1d
    l4 = pp - (1.1 / 2) * range_1d
    
    # Align daily levels to 4h timeframe (daily values update after daily bar closes)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Trend filter: 20-period EMA on 4h close
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_up = close > ema20
    trend_down = close < ema20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to daily pivot point or trend filter reverses
        if position == 1 and (close[i] <= pp_aligned[i] or not trend_up[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= pp_aligned[i] or not trend_down[i]):
            position = 0
            signals[i] = 0.0
        # Entry conditions with all filters
        elif vol_confirm[i] and trend_up[i] and close[i] > h4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif vol_confirm[i] and trend_down[i] and close[i] < l4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals