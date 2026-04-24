#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA on 4h to confirm breakout strength.
- Entry: Long when price breaks above Camarilla R1 AND 1d EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla S1 AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout (S1 for long, R1 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # For intraday, we use the previous 4h bar's high/low/close as proxy for daily
    # But since we're on 4h timeframe, we need to calculate based on previous day's data
    # We'll use 1d data for proper Camarilla calculation
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous day's high
    prev_low = df_1d['low'].shift(1).values    # previous day's low
    prev_close = df_1d['close'].shift(1).values # previous day's close
    
    # True range for Camarilla
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = prev_close + (range_val * 1.1 / 12)
    S1 = prev_close - (range_val * 1.1 / 12)
    R2 = prev_close + (range_val * 1.1 / 6)
    S2 = prev_close - (range_val * 1.1 / 6)
    R3 = prev_close + (range_val * 1.1 / 4)
    S3 = prev_close - (range_val * 1.1 / 4)
    R4 = prev_close + (range_val * 1.1 / 2)
    S4 = prev_close - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 1d EMA(34) for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 4h volume > 2.0 * 20-period volume MA on 4h
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above R1 AND 1d EMA34 bullish (close > EMA34)
                if curr_high > r1 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S1 AND 1d EMA34 bearish (close < EMA34)
                elif curr_low < s1 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR loss of volume confirmation
            if curr_low < s1 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR loss of volume confirmation
            if curr_high > r1 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0