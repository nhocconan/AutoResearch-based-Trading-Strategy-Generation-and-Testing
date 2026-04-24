#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter, volume confirmation, and ATR-based stoploss.
- Primary timeframe: 12h for lower trade frequency and reduced fee drag.
- HTF: 1w EMA50 for major trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Entry: Long when price breaks above Donchian(20) high AND 1w EMA50 bullish AND volume > 1.5 * 20-period volume MA.
         Short when price breaks below Donchian(20) low AND 1w EMA50 bearish AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Donchian breakout (short for long exit, long for short exit) OR ATR-based stoploss (2.0 * ATR).
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy captures medium-term trends with institutional volume confirmation, works in both bull and bear markets
by only trading in the direction of the weekly trend, and uses ATR stops to manage risk during volatile periods.
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
    
    # Calculate 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_confirm = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, Donchian, and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_confirm[i]:
                # Bullish breakout: price > Donchian high AND 1w EMA50 bullish (close > EMA)
                if curr_high > donch_high[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price < Donchian low AND 1w EMA50 bearish (close < EMA)
                elif curr_low < donch_low[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long exit: Donchian breakout in opposite direction OR ATR stoploss
            if curr_low < donch_low[i] or curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout in opposite direction OR ATR stoploss
            if curr_high > donch_high[i] or curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0