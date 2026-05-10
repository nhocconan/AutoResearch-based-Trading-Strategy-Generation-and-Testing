#!/usr/bin/env python3
# 12h_Adaptive_Camarilla_Trend_Breakout
# Hypothesis: Adaptive position sizing based on volatility (ATR) combined with
# Camarilla R3/S3 breakouts and daily trend filter. Uses volatility-adjusted
# sizing to reduce drawdown in volatile periods while maintaining exposure
# in trending markets. Target: 25-35 trades/year to minimize fee drag.

name = "12h_Adaptive_Camarilla_Trend_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # ATR for volatility (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = np.zeros(n)
    atr_sum = 0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 14:
            atr_sum -= tr[i-14]
        if i >= 13:
            atr[i] = atr_sum / 14
        else:
            atr[i] = np.nan
    
    # Volume spike filter (2x 20-period average)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    volume_spike = volume > (2 * vol_ma)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for EMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility-based position sizing (0.15 to 0.35 range)
        atr_ratio = atr[i] / close[i] if close[i] > 0 else 0
        # Normalize ATR ratio: 0.01 -> 0.35, 0.03 -> 0.15 (inverse relationship)
        vol_factor = np.clip(0.5 - (atr_ratio - 0.01) * 10, 0.3, 1.0)
        base_size = 0.25
        position_size = base_size * vol_factor
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and uptrend
            if (high[i] > R3_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike[i]):
                signals[i] = position_size
                position = 1
            # Short: price breaks below S3 with volume spike and downtrend
            elif (low[i] < S3_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike[i]):
                signals[i] = -position_size
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 or trend changes
            if (low[i] < S3_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        
        elif position == -1:
            # Exit: price breaks above R3 or trend changes
            if (high[i] > R3_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals