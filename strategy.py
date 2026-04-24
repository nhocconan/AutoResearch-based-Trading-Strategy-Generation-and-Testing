#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Williams %R(14) < -80 (oversold) AND price > 1w EMA50 AND volume > 1.5 * 12h volume MA(20);
         Short when Williams %R(14) > -20 (overbought) AND price < 1w EMA50 AND volume > 1.5 * 12h volume MA(20).
- Exit: Close above/below opposite Williams %R threshold (-20 for long, -80 for short) for profit-taking,
        with ATR-based stoploss (2.5 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Williams %R for mean reversion extremes, 1w EMA50 trend filter to avoid counter-trend trades,
  and volume confirmation for participation. Designed to work in both bull and bear markets.
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
    
    # Get 12h data for volume MA(20) and ATR(14)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR(14) for 12h timeframe
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 12h timeframe
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Williams %R(14) and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R(14) for 1w timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w Williams %R and EMA50 to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_12h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 1.5x threshold for balanced entry
        vol_confirm = curr_volume > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend)
                if williams_r_aligned[i] < -80 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend)
                elif williams_r_aligned[i] > -20 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.5 * ATR below entry
            stoploss = entry_price - 2.5 * curr_atr
            # Profit take: Williams %R > -20 (exit overbought)
            if curr_close < stoploss or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.5 * ATR above entry
            stoploss = entry_price + 2.5 * curr_atr
            # Profit take: Williams %R < -80 (exit oversold)
            if curr_close > stoploss or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanRev_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0