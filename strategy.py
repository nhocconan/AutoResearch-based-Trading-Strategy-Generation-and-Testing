#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_breakout_volume_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels (H4, L4) for breakout entries in trending markets. Uses 1w HTF for trend alignment (price > weekly EMA20 = uptrend, < = downtrend). Volume confirmation (>1.5x 20-day average) filters weak breakouts. Exits on opposite Camarilla level touch (H4 for longs, L4 for shorts). Designed for low trade frequency (target: 7-25/year) to minimize fee drag, works in bull/bear by following institutional volume-driven breakouts in HTF-aligned trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Camarilla pivot levels (based on prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Typical price for Camarilla calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly range
    weekly_range = df_1w['high'] - df_1w['low']
    # Camarilla levels: H4 = close + 1.5 * range, L4 = close - 1.5 * range
    camarilla_h4 = df_1w['close'] + 1.5 * weekly_range
    camarilla_l4 = df_1w['close'] - 1.5 * weekly_range
    # Align to 1d timeframe (wait for weekly close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4.values)
    
    # Weekly trend filter: EMA20 on weekly close
    weekly_ema20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(weekly_ema20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches or breaks below L4 (opposite level)
            if low[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches or breaks above H4 (opposite level)
            if high[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and HTF trend alignment
            if volume_confirmed:
                # Weekly uptrend: price > weekly EMA20
                if close[i] > weekly_ema20_aligned[i]:
                    # Long: price breaks above H4
                    if high[i] > camarilla_h4_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                # Weekly downtrend: price < weekly EMA20
                elif close[i] < weekly_ema20_aligned[i]:
                    # Short: price breaks below L4
                    if low[i] < camarilla_l4_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals