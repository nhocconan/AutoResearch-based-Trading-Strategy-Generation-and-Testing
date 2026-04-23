#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below lower Donchian AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price retraces to midpoint of Donchian channel or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag. Targets 12-37 trades/year per symbol.
Donchian channels provide robust trend-following structure; EMA50 filters for higher-timeframe trend;
volume confirmation ensures breakout conviction. Works in bull markets (breakouts with volume) and
bear markets (breakdowns with volume). The 12h timeframe reduces trade frequency to avoid fee drag
while still capturing significant moves in BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels from 12h OHLC
    # We need to calculate Donchian on the primary timeframe (12h)
    # But we don't have direct access to 12h OHLC, so we'll approximate using rolling window
    # Since timeframe is 12h, we use 20-period lookback on 12h data
    donchian_window = 20
    upper_dc = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    midpoint_dc = (upper_dc + lower_dc) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_window, 20)  # EMA50 needs 50, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(midpoint_dc[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        upper_dc_val = upper_dc[i]
        lower_dc_val = lower_dc[i]
        midpoint_dc_val = midpoint_dc[i]
        
        if position == 0:
            # Long: Break above upper Donchian AND uptrend (close > EMA50) AND volume spike
            if close[i] > upper_dc_val and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below lower Donchian AND downtrend (close < EMA50) AND volume spike
            elif close[i] < lower_dc_val and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to midpoint of Donchian channel
            if position == 1 and close[i] <= midpoint_dc_val:
                exit_signal = True
            elif position == -1 and close[i] >= midpoint_dc_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0