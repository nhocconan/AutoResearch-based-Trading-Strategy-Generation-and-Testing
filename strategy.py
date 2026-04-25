#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + HMA(21) Trend + Volume Spike + ATR Stop
Hypothesis: Donchian breakouts capture strong momentum; HMA21 filter ensures trend alignment;
volume spike confirms institutional interest; ATR-based stop manages risk. Works in bull/bear
by following 4h trend via HMA. Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(close, period):
    """Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    if half_period == 0 or sqrt_period == 0:
        return close.copy()
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_series = pd.Series(close)
    wma_half = wma(close_series.values, half_period)
    wma_full = wma(close_series.values, period)
    
    # Handle array lengths
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad to original length
    hma_padded = np.full_like(close, np.nan, dtype=float)
    start_idx = period - 1
    end_idx = start_idx + len(hma)
    if end_idx <= len(close):
        hma_padded[start_idx:end_idx] = hma
    
    return hma_padded

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for HTF trend (though primary is 4h, we use 1d for stronger trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for stronger trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # HMA21 on 4h for entry timing
    hma_21 = calculate_hma(close, 21)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # ATR for stoploss (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 21, 20, 14)  # Donchian, HMA, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND bullish bias AND volume spike
            long_entry = (curr_high > highest_high[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian lower AND bearish bias AND volume spike
            short_entry = (curr_low < lowest_low[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * atr[i]
            # Exit: price falls below stop OR breaks below Donchian lower (mean reversion)
            if curr_low < stop_price or curr_close < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * atr[i]
            # Exit: price rises above stop OR breaks above Donchian upper (mean reversion)
            if curr_high > stop_price or curr_close > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HMA21_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0