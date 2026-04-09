#!/usr/bin/env python3
# mtf_1h_camarilla_4h1d_volume_v1
# Hypothesis: 1h strategy using 4h/1d Camarilla pivot confluence for structure, volume confirmation for timing, and session filter (08-20 UTC) to reduce noise.
# Uses 4h trend filter (close > EMA20) for long bias, < EMA20 for short bias. Entries only when price breaks H3/L3 with volume > 2x 20-period MA.
# Exits on close < L3 (long) or close > H3 (short). Position size: ±0.20 to control drawdown and enable discrete levels.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag. Works in bull (breakouts) and bear (mean reversion at extremes).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar: use same day's data (no look-ahead)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance levels
    H3 = pivot + (range_ * 1.1 / 4)
    H4 = pivot + (range_ * 1.1 / 2)
    # Support levels
    L3 = pivot - (range_ * 1.1 / 4)
    L4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR 4h trend turns bearish
            if close[i] < L3_aligned[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR 4h trend turns bullish
            if close[i] > H3_aligned[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Need volume confirmation and session is already filtered above
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # 4h trend filter: only long in uptrend, short in downtrend
                if close[i] > ema_4h_aligned[i]:
                    # Long: price above pivot + close > H3 (breakout)
                    if close[i] > pivot_aligned[i] and close[i] > H3_aligned[i]:
                        position = 1
                        signals[i] = 0.20
                else:
                    # Short: price below pivot + close < L3 (breakdown)
                    if close[i] < pivot_aligned[i] and close[i] < L3_aligned[i]:
                        position = -1
                        signals[i] = -0.20
    
    return signals