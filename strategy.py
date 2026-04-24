#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Bull Power > 0 AND price > 1w EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when Bear Power < 0 AND price < 1w EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Close below/above 13-period EMA on 6h for profit-taking, with ATR-based stoploss (2.5 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Elder Ray to measure bull/bear power relative to EMA13, 1w EMA50 trend filter to avoid counter-trend trades,
  and volume confirmation for participation. Designed to work in both bull and bear markets via trend filter.
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
    
    # Get 6h data for EMA13, EMA50, ATR(14), and volume MA(20)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA13 for 6h timeframe (Elder Ray core)
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate EMA50 for 6h timeframe (additional trend filter)
    ema50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for 6h timeframe
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_6h[0] - low_6h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 6h timeframe
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Calculate Elder Ray components
        bull_power = curr_high - ema13_6h[i]  # Bull Power: High - EMA13
        bear_power = curr_low - ema13_6h[i]   # Bear Power: Low - EMA13
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Bull Power > 0 AND price > 1w EMA50 (uptrend)
                if bull_power > 0 and curr_close > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Bear Power < 0 AND price < 1w EMA50 (downtrend)
                elif bear_power < 0 and curr_close < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.5 * ATR below entry
            stoploss = entry_price - 2.5 * curr_atr
            # Profit take: close below 13-period EMA on 6h
            if curr_close < stoploss or curr_close < ema13_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.5 * ATR above entry
            stoploss = entry_price + 2.5 * curr_atr
            # Profit take: close above 13-period EMA on 6h
            if curr_close > stoploss or curr_close > ema13_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0