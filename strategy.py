#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and 1d weekly pivot confirmation.
- Primary timeframe: 6h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Additional HTF: 1w Camarilla pivot levels (R3/S3 for continuation, R4/S4 for breakout) to filter entries.
- Entry: Price breaks above/below 6h Donchian(20) levels with volume > 1.5 * 20-period volume MA, 1d EMA50 alignment, and price not beyond weekly R4/S4 (avoid overextended breakouts).
- Exit: ATR-based stoploss (2.0 * ATR(14)) or Donchian level reversal (touch opposite level).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by following higher timeframe trend while using lower timeframe breakouts for entry timing, with weekly pivot structure to avoid false breakouts in overextended conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_camarilla_pivots

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian levels and ATR
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 6h Donchian levels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Upper band: highest high of last 20 periods
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    
    # Calculate 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate weekly Camarilla pivots (using previous week's high/low/close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Compute Camarilla levels for each week
    camarilla_levels = compute_camarilla_pivots(high_1w, low_1w, close_1w)
    # Extract R3, R4, S3, S4 levels
    r3 = camarilla_levels[:, 0]  # R3 level
    r4 = camarilla_levels[:, 1]  # R4 level
    s3 = camarilla_levels[:, 2]  # S3 level
    s4 = camarilla_levels[:, 3]  # S4 level
    
    # Align weekly Camarilla levels to 6h timeframe (with 1-bar delay for weekly close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3, additional_delay_bars=1)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3, additional_delay_bars=1)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_6h_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
            
            # Determine 1d EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
            trend_bullish = close[i] > ema_50_aligned[i]
            trend_bearish = close[i] < ema_50_aligned[i]
            
            # Avoid overextended breakouts: price should not be beyond weekly R4/S4
            not_overextended_long = curr_close < r4_aligned[i]
            not_overextended_short = curr_close > s4_aligned[i]
            
            # Long: price breaks above Donchian upper level AND 1d trend bullish AND volume confirmed AND not overextended
            if curr_high > upper_20_aligned[i] and trend_bullish and vol_confirmed and not_overextended_long:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower level AND 1d trend bearish AND volume confirmed AND not overextended
            elif curr_low < lower_20_aligned[i] and trend_bearish and vol_confirmed and not_overextended_short:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Donchian lower level (reversal signal)
            stop_loss = entry_price - 2.0 * atr_6h_aligned[i]
            if curr_low < stop_loss or curr_low < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Donchian upper level (reversal signal)
            stop_loss = entry_price + 2.0 * atr_6h_aligned[i]
            if curr_high > stop_loss or curr_high > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_1wCamarilla_v1"
timeframe = "6h"
leverage = 1.0