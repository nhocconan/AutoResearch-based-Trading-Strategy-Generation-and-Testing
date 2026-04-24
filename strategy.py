#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend direction (price > EMA34 = bullish, price < EMA34 = bearish).
- Camarilla pivot levels (H3, L3) calculated from prior 1d OHLC: 
  H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4.
- Entry: Long when price > H3 AND bullish trend AND volume > 2.0 * 20-period average volume.
         Short when price < L3 AND bearish trend AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla breakout (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in alignment with 1d trend,
  avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate prior 1d OHLC for Camarilla levels (H3, L3)
    # We need the completed 1d bar's OHLC, so we shift by 1 to avoid look-ahead
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior completed 1d bar OHLC (shifted by 1 to ensure we only use closed bars)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate Camarilla H3 and L3 levels from prior 1d bar
    range_1d = high_1d_prev - low_1d_prev
    camarilla_h3_1d = close_1d_prev + 1.1 * range_1d / 4
    camarilla_l3_1d = close_1d_prev - 1.1 * range_1d / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: bullish if price > EMA34, bearish if price < EMA34
        bullish_trend = curr_close > ema34_aligned[i]
        bearish_trend = curr_close < ema34_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price < H3
            if position == 1:
                if curr_close < camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3
            elif position == -1:
                if curr_close > camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > H3 AND bullish trend AND volume confirmation
            long_condition = (curr_close > camarilla_h3_aligned[i] and 
                            bullish_trend and
                            volume_confirm)
            
            # Short: price < L3 AND bearish trend AND volume confirmation
            short_condition = (curr_close < camarilla_l3_aligned[i] and 
                             bearish_trend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_CamarillaH3L3_Breakout_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0