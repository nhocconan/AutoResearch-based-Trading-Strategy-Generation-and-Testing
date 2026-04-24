#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter.
- Entry: Long when price breaks above Donchian(20) upper band AND price > 1d EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian(20) lower band AND price < 1d EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via ATR trailing (implemented as signal=0 when price closes below/above ATR-based trailing stop).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian channels provide clear structure; 1d EMA50 filters counter-trend breakouts in bear markets; volume spike confirms conviction.
- Works in both bull (trend continuation) and bear (mean reversion via tight stops) regimes.
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
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 14, 20)  # 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Calculate dynamic trailing stop
        if position == 1:
            # Long: trail below highest high since entry by 2.5 * ATR
            if i == start_idx or position == 0:
                long_stop = highest_high[i] - 2.5 * atr[i]
            else:
                long_stop = max(long_stop, highest_high[i] - 2.5 * atr[i])
            
            if curr_close < long_stop:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            # Short: trail above lowest low since entry by 2.5 * ATR
            if i == start_idx or position == 0:
                short_stop = lowest_low[i] + 2.5 * atr[i]
            else:
                short_stop = min(short_stop, lowest_low[i] + 2.5 * atr[i])
            
            if curr_close > short_stop:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > highest_high[i]
        bearish_breakout = curr_close < lowest_low[i]
        
        # Trend filter from 1d EMA50
        price_above_ema = curr_close > ema_50_aligned[i]
        price_below_ema = curr_close < ema_50_aligned[i]
        
        # Volume confirmation (spike)
        vol_confirm = curr_volume > 2.0 * volume_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: bullish breakout AND price above 1d EMA50
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.30
                    position = 1
                    long_stop = highest_high[i] - 2.5 * atr[i]  # initialize stop
                # Short: bearish breakout AND price below 1d EMA50
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.30
                    position = -1
                    short_stop = lowest_low[i] + 2.5 * atr[i]  # initialize stop
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_EMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0