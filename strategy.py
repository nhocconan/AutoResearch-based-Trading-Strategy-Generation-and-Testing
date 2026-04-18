#!/usr/bin/env python3
"""
6h_OrderBlock_With_OrderFlow_And_TrendFilter
Hypothesis: Institutional order blocks (OB) identified via volume imbalances on 6h chart,
filtered by 1d trend (EMA50) and order flow imbalance (OFI) to avoid false breakouts.
Works in both bull/bear by trading with trend direction. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Order Block detection: bullish OB = down candle with high volume followed by up candle
    # Bearish OB = up candle with high volume followed by down candle
    # Volume imbalance: current candle volume > 1.5 * average of last 20 candles
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Avoid division by zero handled by min_periods
    
    # Bullish OB: red candle (close < open) with high volume, next candle closes above its high
    bearish_candle = close < prices['open'].values
    bullish_ob_signal = bearish_candle & (vol_ratio > 1.5)
    bullish_ob_high = np.where(bullish_ob_signal, high, np.nan)
    # Forward fill to get OB zone until broken
    bullish_ob_level = pd.Series(bullish_ob_high).ffill().values
    
    # Bearish OB: green candle (close > open) with high volume, next candle closes below its low
    bullish_candle = close > prices['open'].values
    bearish_ob_signal = bullish_candle & (vol_ratio > 1.5)
    bearish_ob_low = np.where(bearish_ob_signal, low, np.nan)
    bearish_ob_level = pd.Series(bearish_ob_low).ffill().values
    
    # Order Flow Imbalance: (buy volume - sell volume) / total volume
    # Using proxy: (close - low) / (high - low) for buying pressure
    # Avoid division by zero
    hl_range = high - low
    buying_pressure = np.where(hl_range > 0, (close - low) / hl_range, 0.5)
    # Smooth to get OFI signal
    ofi = pd.Series(buying_pressure).ewm(span=10, adjust=False, min_periods=10).mean().values
    # OFI > 0.55 = buying pressure, < 0.45 = selling pressure
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20, 10)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bullish_ob_level[i]) or
            np.isnan(bearish_ob_level[i]) or
            np.isnan(ofi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema_50_aligned[i]
        bull_ob = bullish_ob_level[i]
        bear_ob = bearish_ob_level[i]
        ofi_val = ofi[i]
        
        if position == 0:
            # Long: price breaks above bullish OB with buying pressure and uptrend
            if price > bull_ob and ofi_val > 0.55 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish OB with selling pressure and downtrend
            elif price < bear_ob and ofi_val < 0.45 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below bullish OB OR trend turns down
            if price < bull_ob or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above bearish OB OR trend turns up
            if price > bear_ob or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_OrderBlock_With_OrderFlow_And_TrendFilter"
timeframe = "6h"
leverage = 1.0