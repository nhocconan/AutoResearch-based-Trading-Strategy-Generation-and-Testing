#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above H3 level AND price > 1w EMA50 AND volume > 1.5 * 1d volume MA(20);
         Short when price breaks below L3 level AND price < 1w EMA50 AND volume > 1.5 * 1d volume MA(20).
- Exit: Close below/above L3/H3 levels for mean reversion, with ATR-based stoploss (2.0 * ATR(14)).
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels for intraday reversal points, volume confirmation for participation,
  1w EMA50 trend filter to avoid counter-trend trades, and ATR for risk management.
- Designed to work in both bull and bear markets via trend filter and mean-reversion structure.
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
    
    # Get 1d data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    h3_level = prev_close + (prev_range * 1.1 / 4)
    l3_level = prev_close - (prev_range * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (already aligned, but ensure proper shifting)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_level)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for 1d timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 1d timeframe
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_1d[i])):
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
        
        # Volume confirmation: 1.5x threshold for balanced entry
        vol_confirm = curr_volume > 1.5 * vol_ma_1d[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above H3 AND price > 1w EMA50 (uptrend)
                if curr_close > h3_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: price breaks below L3 AND price < 1w EMA50 (downtrend)
                elif curr_close < l3_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr
            # Mean reversion exit: close below L3 level
            if curr_close < stoploss or curr_close < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr
            # Mean reversion exit: close above H3 level
            if curr_close > stoploss or curr_close > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0