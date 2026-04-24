#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based position sizing.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Price breaks above/below 4h Donchian(20) channel with 1d EMA34 alignment.
- Exit: ATR-based stoploss (2.0 * ATR(14)) or Donchian channel reversal (touch opposite level).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by following higher timeframe trend while using lower timeframe breakouts for entry timing.
ATR-based position sizing adjusts exposure based on volatility to manage drawdown.
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
    
    # Get 4h data for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels (based on previous 20 bars)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper channel: highest high of previous 20 bars
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower channel: lowest low of previous 20 bars
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (already on 4h, but ensure proper alignment)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h ATR(14) for stoploss and position sizing
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with 1d EMA34 trend filter
            trend_bullish = close[i] > ema_34_aligned[i]
            trend_bearish = close[i] < ema_34_aligned[i]
            
            # Long: price breaks above Donchian upper channel AND 1d trend bullish
            if curr_high > donchian_high_aligned[i] and trend_bullish:
                # ATR-based position sizing: 0.25 * (ATR(14) / close) scaled to max 0.30
                vol_factor = min(0.30, 0.25 * (atr_4h_aligned[i] / curr_close) * 100)
                signals[i] = vol_factor
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian lower channel AND 1d trend bearish
            elif curr_low < donchian_low_aligned[i] and trend_bearish:
                # ATR-based position sizing: 0.25 * (ATR(14) / close) scaled to max 0.30
                vol_factor = min(0.30, 0.25 * (atr_4h_aligned[i] / curr_close) * 100)
                signals[i] = -vol_factor
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Donchian lower channel (reversal signal)
            stop_loss = entry_price - 2.0 * atr_4h_aligned[i]
            if curr_low < stop_loss or curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = vol_factor  # maintain current position size
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Donchian upper channel (reversal signal)
            stop_loss = entry_price + 2.0 * atr_4h_aligned[i]
            if curr_high > stop_loss or curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -vol_factor  # maintain current position size
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_ATR_PositionSizing_v1"
timeframe = "4h"
leverage = 1.0