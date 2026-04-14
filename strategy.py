#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_open = df_1d['open'].values
    
    # ATR(14) on daily
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.full_like(tr, np.nan, dtype=float)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Daily trend: 1 if bullish (close > open), -1 if bearish
    daily_bullish = (daily_close > daily_open).astype(float)
    daily_bearish = (daily_close < daily_open).astype(float)
    
    # Bollinger Bands (20, 2) on daily close
    sma_20 = np.full_like(daily_close, np.nan)
    std_20 = np.full_like(daily_close, np.nan)
    for i in range(19, len(daily_close)):
        sma_20[i] = np.mean(daily_close[i-19:i+1])
        std_20[i] = np.std(daily_close[i-19:i+1])
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Bollinger Band Width for regime detection
    bb_width = (upper_band - lower_band) / sma_20
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(49, len(bb_width)):  # 50-period lookback
        window = bb_width[i-49:i+1]
        if not np.all(np.isnan(window)):
            rank = np.sum(~np.isnan(window) & (bb_width[i] >= window)) / np.sum(~np.isnan(window))
            bb_width_percentile[i] = rank
    
    # Align daily indicators to 12h
    daily_bull_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bear_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any indicator not ready
        if (np.isnan(daily_bull_aligned[i]) or np.isnan(daily_bear_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(atr_14_aligned[i])):
            continue
        
        bb_percentile = bb_width_percentile_aligned[i]
        daily_bull = daily_bull_aligned[i]
        daily_bear = daily_bear_aligned[i]
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: Bollinger squeeze breakout above upper band in bullish daily trend
            if (bb_percentile < 0.2 and  # Squeeze: low volatility
                close[i] > upper_band[int(np.sum(~np.isnan(daily_close) & (np.arange(len(daily_close)) <= i)))-1] if i < len(daily_close) else upper_band[-1] and
                daily_bull > 0.5 and
                volume[i] > np.nanmedian(volume[max(0, i-20):i+1]) * 1.5):  # Volume spike
                position = 1
                signals[i] = position_size
            # Short: Bollinger squeeze breakout below lower band in bearish daily trend
            elif (bb_percentile < 0.2 and  # Squeeze: low volatility
                  close[i] < lower_band[int(np.sum(~np.isnan(daily_low) & (np.arange(len(daily_low)) <= i)))-1] if i < len(daily_low) else lower_band[-1] and
                  daily_bear > 0.5 and
                  volume[i] > np.nanmedian(volume[max(0, i-20):i+1]) * 1.5):  # Volume spike
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price closes below Bollinger middle or volatility expands
            sma_val = sma_20[int(np.sum(~np.isnan(daily_close) & (np.arange(len(daily_close)) <= i)))-1] if i < len(daily_close) else sma_20[-1]
            if close[i] < sma_val or bb_percentile > 0.8:  # Mean reversion or high volatility
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price closes above Bollinger middle or volatility expands
            sma_val = sma_20[int(np.sum(~np.isnan(daily_close) & (np.arange(len(daily_close)) <= i)))-1] if i < len(daily_close) else sma_20[-1]
            if close[i] > sma_val or bb_percentile > 0.8:  # Mean reversion or high volatility
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Bollinger_Squeeze_Breakout_DailyTrend"
timeframe = "12h"
leverage = 1.0