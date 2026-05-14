#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend filter + volume spike confirmation
# Uses discrete sizing 0.30 to balance return and risk. Target: 75-200 trades over 4 years (19-50/year).
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out by 12h HMA).
# Focus on BTC/ETH as primary symbols with proven edge from Donchian + volume + trend confluence.
# Uses 12h timeframe for HTF as specified in experiment.

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h HMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    hma_21_12h = calculate_hma(df_12h['close'].values, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 21, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_hma_21_12h = hma_21_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 12h HMA trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above Donchian high + price above 12h HMA21
                if curr_close > curr_donchian_high and curr_close > curr_hma_21_12h:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Donchian low + price below 12h HMA21
                elif curr_close < curr_donchian_low and curr_close < curr_hma_21_12h:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below Donchian low OR loses 12h trend
            if curr_low <= stop_loss or curr_close < curr_donchian_low or curr_close < curr_hma_21_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above Donchian high OR loses 12h trend
            if curr_high >= stop_loss or curr_close > curr_donchian_high or curr_close > curr_hma_21_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    # WMA of full period
    wma_full = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    # Raw HMA = 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA = WMA(sqrt_period) of raw_hma
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    return hma