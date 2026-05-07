#!/usr/bin/env python3
name = "4h_4hTrend_1dEngulfingVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h trend: 4h EMA(21)
    ema_21_4h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily bullish engulfing pattern: current candle engulfs previous bearish candle
    # Engulfing condition: today's close > yesterday's open AND today's open < yesterday's close
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    bullish_engulf = (daily_close[1:] > daily_open[:-1]) & (daily_open[1:] < daily_close[:-1])
    bearish_engulf = (daily_close[1:] < daily_open[:-1]) & (daily_open[1:] > daily_close[:-1])
    
    # Align to 4h timeframe - daily patterns available after daily close
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[False], bullish_engulf]))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, np.concatenate([[False], bearish_engulf]))
    
    # Volume spike: 4h volume > 2x 24-period average (24*4h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_4h[i]) or np.isnan(bullish_engulf_aligned[i]) or 
            np.isnan(bearish_engulf_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing on daily + price above 4h EMA + volume spike
            if (bullish_engulf_aligned[i] and 
                close[i] > ema_21_4h[i] and 
                volume[i] > vol_ma_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing on daily + price below 4h EMA + volume spike
            elif (bearish_engulf_aligned[i] and 
                  close[i] < ema_21_4h[i] and 
                  volume[i] > vol_ma_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below 4h EMA or volume drops
            if close[i] < ema_21_4h[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above 4h EMA or volume drops
            if close[i] > ema_21_4h[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h trend filtered by daily candlestick engulfing patterns with volume confirmation
# - 4h EMA(21) establishes trend direction (price above = uptrend, below = downtrend)
# - Daily bullish/bearish engulfing patterns signal potential reversals/continuations
# - Volume spike (2x average) confirms institutional participation in the move
# - Works in bull markets: buy on bullish engulfing dips in uptrend
# - Works in bear markets: sell on bearish engulfing rallies in downtrend
# - Exit when price crosses 4h EMA or volume weakens
# - Position size 0.25 targets ~20-40 trades/year, avoiding excessive fee drag
# - Daily engulfing provides clean reversal signals that work across market regimes
# - Combines trend following with reversal confirmation for robustness