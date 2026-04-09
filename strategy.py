#!/usr/bin/env python3
# 1h_4h1d_camarilla_pivot_v1
# Hypothesis: 1h Camarilla pivot breakouts with 4h/1d trend filter and volume confirmation.
# Works in bull/bear: 4h/1d EMA50 defines institutional trend; 1h Camarilla levels (H3/L3) act as
# magnet zones where breakouts capture momentum with institutional participation.
# Volume confirms breakout legitimacy. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_camarilla_pivot_v1"
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
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d HTF data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    # Camarilla levels based on previous day's range
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    
    # Calculate Camarilla levels for each 1d bar
    # H3 = Close + (High - Low) * 1.1/4
    # L3 = Close - (High - Low) * 1.1/4
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla H3 OR trend turns bearish
            if close[i] < camarilla_h3_aligned[i] or close[i] < ema50_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla L3 OR trend turns bullish
            if close[i] > camarilla_l3_aligned[i] or close[i] > ema50_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above Camarilla H3 with bullish trend (both timeframes)
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema50_4h_aligned[i] and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L3 with bearish trend (both timeframes)
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema50_4h_aligned[i] and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals