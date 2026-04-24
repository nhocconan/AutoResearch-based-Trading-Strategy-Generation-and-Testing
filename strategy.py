#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 1.8 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above H3 level AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below L3 level AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Camarilla level touch (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Camarilla pivot levels provide high-probability reversal/breakout points, and combining with 1d trend
and volume confirmation reduces false signals while capturing strong moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels use the previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close as previous
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels calculation
    range_ = prev_high - prev_low
    h3 = prev_close + range_ * 1.1 / 4
    l3 = prev_close - range_ * 1.1 / 4
    h4 = prev_close + range_ * 1.1 / 2
    l4 = prev_close - range_ * 1.1 / 2
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume MA for volume confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above H3 AND 1d EMA34 bullish (close > EMA34)
                if curr_high > h3[i] and close[i] > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below L3 AND 1d EMA34 bearish (close < EMA34)
                elif curr_low < l3[i] and close[i] < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches L3 level OR loss of volume confirmation
            if curr_low <= l3[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches H3 level OR loss of volume confirmation
            if curr_high >= h3[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0