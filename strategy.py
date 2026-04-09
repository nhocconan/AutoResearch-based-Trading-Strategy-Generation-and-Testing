#!/usr/bin/env python3
# 6h_camarilla_pivot_breakout_volume_v1
# Hypothesis: 6h strategy using Camarilla pivot levels from 1d HTF. Fade at R3/S3 levels (mean reversion) with volume confirmation (>1.3x 20-bar avg volume), breakout continuation at R4/S4 levels with volume and trend filter (price > 6h EMA50 for longs, < EMA50 for shorts). Works in bull/bear: mean reversion in ranging markets, breakout continuation in trending markets, volume confirms conviction. Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_breakout_volume_v1"
timeframe = "6h"
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
    
    # 6h EMA(50) for trend filter on breakouts
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Multi-timeframe: 1d OHLC for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous 1d bar)
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_high = high_1d
    camarilla_low = low_1d
    camarilla_close = close_1d
    camarilla_range = camarilla_high - camarilla_low
    
    r4 = camarilla_close + 1.5 * camarilla_range
    r3 = camarilla_close + 1.1 * camarilla_range
    s3 = camarilla_close - 1.1 * camarilla_range
    s4 = camarilla_close - 1.5 * camarilla_range
    
    # Align 1d Camarilla levels to 6h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(ema_50[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below R3 (fade level) or stoploss via opposite signal
            if close[i] < r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 (fade level) or stoploss via opposite signal
            if close[i] > s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade at R3/S3 (mean reversion) with volume confirmation
            fade_long = (close[i] <= r3_aligned[i]) and volume_confirmed
            fade_short = (close[i] >= s3_aligned[i]) and volume_confirmed
            
            # Breakout continuation at R4/S4 with volume and trend filter
            breakout_long = (close[i] >= r4_aligned[i]) and volume_confirmed and (close[i] > ema_50[i])
            breakout_short = (close[i] <= s4_aligned[i]) and volume_confirmed and (close[i] < ema_50[i])
            
            if fade_long:
                position = 1
                signals[i] = 0.25
            elif fade_short:
                position = -1
                signals[i] = -0.25
            elif breakout_long:
                position = 1
                signals[i] = 0.25
            elif breakout_short:
                position = -1
                signals[i] = -0.25
    
    return signals