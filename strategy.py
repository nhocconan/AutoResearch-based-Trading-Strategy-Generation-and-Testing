#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze + 12h ADX Trend Strength + Volume Confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for ADX trend filter.
- Entry: Bollinger Band Width at 20-period low (squeeze) + close breaks above upper band (long) or below lower band (short) on 6h close, with volume > 1.5x 20-period volume MA.
- Direction filter: only long when 12h ADX(14) > 25 (trending market), only short when 12h ADX(14) > 25.
- Bollinger Squeeze identifies low volatility primed for breakout; ADX ensures we only trade strong trends.
- Volume confirmation reduces false breakouts.
- Exit: opposite Bollinger Band touch (long exits at lower band, short exits at upper band) or ADX < 20 (trend weakens).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrends, in bear via selling breakdowns in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h ADX(14) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[up_move < 0] = 0
    down_move[down_move < 0] = 0
    
    # Smoothed values
    def _wilder_smooth(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period = 14
    tr_smooth = _wilder_smooth(tr, period)
    up_smooth = _wilder_smooth(up_move, period)
    down_smooth = _wilder_smooth(down_move, period)
    
    # DI+ and DI-
    plus_di = 100 * up_smooth / tr_smooth
    minus_di = 100 * down_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    mask = (plus_di + minus_di) != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    adx = _wilder_smooth(dx, period)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Bollinger Bands on 6h data
    if len(close) < 20:
        return np.zeros(n)
    
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2.0 * std_20)
    lower_band = sma_20 - (2.0 * std_20)
    
    # Bollinger Band Width (for squeeze detection)
    bb_width = (upper_band - lower_band) / sma_20
    # 20-period low of BB Width (squeeze condition)
    bb_width_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width <= bb_width_low
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14+14+14, 20, 20)  # 12h ADX needs 14+14+14, BBands needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bb_squeeze[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger squeeze + break above upper band + volume spike + ADX > 25 (strong trend)
            if (bb_squeeze[i] and close[i] > upper_band[i] and volume_spike[i] and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze + break below lower band + volume spike + ADX > 25 (strong trend)
            elif (bb_squeeze[i] and close[i] < lower_band[i] and volume_spike[i] and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to lower band or ADX < 20 (trend weakens)
            if close[i] < lower_band[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to upper band or ADX < 20 (trend weakens)
            if close[i] > upper_band[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_12hADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0