#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA(34) trend filter and 12h volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 2.0 * 20-period 12h volume MA to avoid false breakouts.
- Entry: Long when price breaks above Camarilla H3 AND 1d EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla L3 AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    # For intraday, we use previous 12h bar's OHLC to calculate levels for current 12h bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4.0)  # resistance level 3
    L3 = pivot - (range_hl * 1.1 / 4.0)  # support level 3
    
    # Get 1d data for EMA(34) trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA
    volume_spike = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need enough 1d bars for EMA34 and 20 periods for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(H3[i]) or 
            np.isnan(L3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        camarilla_H3 = H3[i]
        camarilla_L3 = L3[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above Camarilla H3 AND 1d EMA34 bullish (close > EMA34)
                if curr_high > camarilla_H3 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Camarilla L3 AND 1d EMA34 bearish (close < EMA34)
                elif curr_low < camarilla_L3 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla L3 OR loss of volume confirmation
            if curr_low < camarilla_L3 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla H3 OR loss of volume confirmation
            if curr_high > camarilla_H3 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0