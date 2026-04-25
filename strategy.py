#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrendFilter_SessionVolume
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA20 trend filter, volume spike, and session filter (08-20 UTC). 
In 4h-trending markets (price > EMA20 for longs, price < EMA20 for shorts), breakouts have higher success. 
Volume confirms breakout validity. Session filter avoids low-liquidity hours. Discrete sizing (0.20) minimizes fee churn. 
Target: 15-35 trades/year, works in both bull/bear by following 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA20 on 4h close for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels on 4h data (based on previous 4h bar's OHLC)
    # Camarilla: R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    camarilla_r1_4h = close_4h + ((high_4h - low_4h) * 1.1 / 12)
    camarilla_s1_4h = close_4h - ((high_4h - low_4h) * 1.1 / 12)
    camarilla_c_4h = close_4h  # Camarilla C is the close
    
    # Align HTF indicators to 1h timeframe (completed 4h bar lag)
    ema20_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h, additional_delay_bars=1)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_4h, camarilla_c_4h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 24-bar average volume (1h TF)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour  # open_time is already datetime64[ms], index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA20 and volume MA
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_c_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 4h trend with volume confirmation
            # Long: price breaks above R1 in uptrend (close > EMA20)
            # Short: price breaks below S1 in downtrend (close < EMA20)
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema20_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema20_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] < camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above Camarilla C (mean reversion to midpoint)
            exit_signal = close[i] > camarilla_c_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrendFilter_SessionVolume"
timeframe = "1h"
leverage = 1.0