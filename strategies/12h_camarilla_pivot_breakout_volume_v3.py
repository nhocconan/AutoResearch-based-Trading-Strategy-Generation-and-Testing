#!/usr/bin/env python3
# 12h_camarilla_pivot_breakout_volume_v3
# Hypothesis: 12h Camarilla pivot breakout with volume confirmation (>1.3x 20-period average) and 1d HTF trend filter (price > 50-period EMA). Enters long when price breaks above H3 level with volume confirmation and bullish 1d trend; short when price breaks below L3 level with volume confirmation and bearish 1d trend. Exits on opposite Camarilla level touch (L3 for long, H3 for short). Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 12-37 trades/year) to work in both bull and bear markets by following institutional volume-driven breakouts in alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_breakout_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + 1.5*(H-L), H3 = C + 1.0*(H-L), H2 = C + 0.5*(H-L), H1 = C + 0.25*(H-L)
    # L1 = C - 0.25*(H-L), L2 = C - 0.5*(H-L), L3 = C - 1.0*(H-L), L4 = C - 1.5*(H-L)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # True range for the day
    day_range = prev_high - prev_low
    
    # Camarilla levels
    H3 = prev_close + 1.0 * day_range
    L3 = prev_close - 1.0 * day_range
    H4 = prev_close + 1.5 * day_range
    L4 = prev_close - 1.5 * day_range
    
    # Align Camarilla levels to 12h timeframe (using previous day's close as anchor)
    # The Camarilla levels are valid for the entire day following the previous day's close
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d HTF trend filter: 50-period EMA on 1d timeframe
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks L3 level (or stoploss at L4)
            if close[i] <= L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks H3 level (or stoploss at H4)
            if close[i] >= H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and 1d trend alignment
            if volume_confirmed:
                # Bullish 1d trend: price above 50-period EMA
                bullish_trend = close[i] > ema_50_1d_aligned[i]
                # Bearish 1d trend: price below 50-period EMA
                bearish_trend = close[i] < ema_50_1d_aligned[i]
                
                # Long: price breaks above H3 level with volume and bullish 1d trend
                if close[i] > H3_aligned[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L3 level with volume and bearish 1d trend
                elif close[i] < L3_aligned[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals