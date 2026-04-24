#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for major trend filter (price > EMA50 = bull regime, price < EMA50 = bear regime).
- Entry: Long when price breaks above Camarilla H3 AND price > 1w EMA50 AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below Camarilla L3 AND price < 1w EMA50 AND volume > 1.5 * 6h volume MA(20).
- Exit: Reversion to Camarilla H4/L4 levels for profit taking, or opposite H3/L3 break for reversal.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot structure from daily timeframe, 1w EMA50 for regime filter,
  and volume confirmation to ensure participation. Designed to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (prior completed day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4/H3/L3/L4 based on prior day range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    daily_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * daily_range
    camarilla_h3 = close_1d + 1.0 * daily_range
    camarilla_l3 = close_1d - 1.0 * daily_range
    camarilla_l4 = close_1d - 1.5 * daily_range
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 1w data for EMA50 trend filter (major regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 6h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry frequency
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Camarilla H3 AND price > 1w EMA50 (bull regime)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Camarilla L3 AND price < 1w EMA50 (bear regime)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: check exit conditions
            # Profit take: reversion to Camarilla H4 (strong resistance)
            # Reverse: break below Camarilla L3 (regime change)
            if curr_close > camarilla_h4_aligned[i] or curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Profit take: reversion to Camarilla L4 (strong support)
            # Reverse: break above Camarilla H3 (regime change)
            if curr_close < camarilla_l4_aligned[i] or curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0