#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + HMA Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum moves. Combined with
Hull Moving Average trend filter and volume confirmation, this strategy works in
both bull (breakouts to new highs) and bear (breakdowns to new lows) markets.
ATR-based stoploss limits drawdown. Designed for 4h timeframe with low trade
frequency (target: 75-200 total trades over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma_2x = pd.Series(series).ewm(span=half, adjust=False, min_periods=half).mean()
    wma = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma_2x - wma
    hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False, min_periods=sqrt).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for HMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 21-period HMA on 1d close for trend
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian (20), HMA, volume MA, ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        hma_trend = hma_21_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND price > HMA (uptrend)
            long_entry = (curr_close > donch_high) and vol_spike and (curr_close > hma_trend)
            # Short: price breaks below Donchian low AND volume spike AND price < HMA (downtrend)
            short_entry = (curr_close < donch_low) and vol_spike and (curr_close < hma_trend)
            
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
            # Exit: price crosses below Donchian low OR ATR-based stoploss hit
            if (curr_close < donch_low) or (curr_close < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high OR ATR-based stoploss hit
            if (curr_close > donch_high) or (curr_close > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HMA21_Trend_VolumeSpike_ATRstop"
timeframe = "4h"
leverage = 1.0