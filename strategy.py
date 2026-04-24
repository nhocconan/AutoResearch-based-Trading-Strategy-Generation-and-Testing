#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR trend filter and volume confirmation.
- Primary timeframe: 12h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for trend filter (bullish if ATR rising, bearish if ATR falling).
- Camarilla levels: Calculate H3, L3, H4, L4 from prior 1d session.
- Entry: Long when close crosses above H3 with volume > 1.5x 20-period average AND 1d ATR rising.
         Short when close crosses below L3 with volume > 1.5x 20-period average AND 1d ATR falling.
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.0*ATR,
        exit short when price > lowest_low_since_entry + 2.0*ATR.
- Signal size: 0.25 discrete to balance return and drawdown.
- Works in both bull and bear markets by using ATR trend (volatility expansion) as regime filter.
- Camarilla levels provide institutional support/resistance with high probability breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for trend filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[:-1])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    # ATR trend: rising if current > previous, falling if current < previous
    atr_rising = np.concatenate([[False], atr_1d[1:] > atr_1d[:-1]])
    atr_falling = np.concatenate([[False], atr_1d[1:] < atr_1d[:-1]])
    
    # Calculate prior 1d Camarilla levels (H3, L3, H4, L4)
    # H4 = Close + 1.5*(High - Low)
    # L4 = Close - 1.5*(High - Low)
    # H3 = Close + 1.125*(High - Low)
    # L3 = Close - 1.125*(High - Low)
    df_1d_range = df_1d_high - df_1d_low
    camarilla_h4 = df_1d_close + 1.5 * df_1d_range
    camarilla_l4 = df_1d_close - 1.5 * df_1d_range
    camarilla_h3 = df_1d_close + 1.125 * df_1d_range
    camarilla_l3 = df_1d_close - 1.125 * df_1d_range
    
    # Align HTF indicators to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising)
    atr_falling_aligned = align_htf_to_ltf(prices, df_1d, atr_falling)
    
    # Calculate 20-period volume average for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need enough bars for ATR and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_rising_aligned[i]) or np.isnan(atr_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            # Long: close crosses above H3 AND volume > 1.5x average AND ATR rising
            if (curr_close > camarilla_h3_aligned[i] and 
                close[i-1] <= camarilla_h3_aligned[i-1] and
                curr_volume > 1.5 * vol_ma[i] and
                atr_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: close crosses below L3 AND volume > 1.5x average AND ATR falling
            elif (curr_close < camarilla_l3_aligned[i] and 
                  close[i-1] >= camarilla_l3_aligned[i-1] and
                  curr_volume > 1.5 * vol_ma[i] and
                  atr_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.0*ATR
            # Calculate current ATR for stoploss (using 14-period ATR on 12h data)
            if i >= 14:
                tr1 = np.abs(high[i] - low[i])
                tr2 = np.abs(high[i] - close[i-1])
                tr3 = np.abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                # Simplified ATR calculation for stop - using recent volatility
                if i >= 28:
                    atr_stop = np.mean([
                        np.abs(high[i-13:i+1] - low[i-13:i+1]),
                        np.abs(high[i-13:i+1] - close[i-14:i]),
                        np.abs(low[i-13:i+1] - close[i-14:i])
                    ]).mean() if hasattr(np.mean([np.abs(high[i-13:i+1] - low[i-13:i+1]), np.abs(high[i-13:i+1] - close[i-14:i]), np.abs(low[i-13:i+1] - close[i-14:i])]), 'mean') else np.abs(high[i] - low[i])
                else:
                    atr_stop = np.abs(high[i] - low[i])
            else:
                atr_stop = np.abs(high[i] - low[i])
            
            if curr_close < highest_since_entry - 2.0 * atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.0*ATR
            if i >= 14:
                tr1 = np.abs(high[i] - low[i])
                tr2 = np.abs(high[i] - close[i-1])
                tr3 = np.abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                if i >= 28:
                    atr_stop = np.mean([
                        np.abs(high[i-13:i+1] - low[i-13:i+1]),
                        np.abs(high[i-13:i+1] - close[i-14:i]),
                        np.abs(low[i-13:i+1] - close[i-14:i])
                    ]).mean() if hasattr(np.mean([np.abs(high[i-13:i+1] - low[i-13:i+1]), np.abs(high[i-13:i+1] - close[i-14:i]), np.abs(low[i-13:i+1] - close[i-14:i])]), 'mean') else np.abs(high[i] - low[i])
                else:
                    atr_stop = np.abs(high[i] - low[i])
            else:
                atr_stop = np.abs(high[i] - low[i])
            
            if curr_close > lowest_since_entry + 2.0 * atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0