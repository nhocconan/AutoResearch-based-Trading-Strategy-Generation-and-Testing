#!/usr/bin/env python3
name = "6h_Engulfing_1dTrend_Threshold"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bullish engulfing: current bullish candle engulfs previous bearish candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close) & \
                     (close > open_price) & (open_price < close)  # Placeholder, will fix below
    # Actually compute bullish and bearish engulfing properly
    bullish_engulf = (close > open_price) & (open_price <= close) & \
                     (close[1:] > open_price[:-1]) & (open_price[1:] <= close[:-1]) & \
                     (close > open_price[:-1]) & (open_price < close[:-1])
    # Fix: shift to align indices
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bullish_engulf[i] = (close[i] > open_price[i]) and (open_price[i] <= close[i]) and \
                            (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        bearish_engulf[i] = (close[i] < open_price[i]) and (open_price[i] >= close[i]) and \
                            (close[i] < open_price[i-1]) and (open_price[i] > close[i-1])
    
    # Align engulfing signals to 60-minute timeframe (no alignment needed as computed on 6h)
    # But we need to ensure we don't use future data - already fine as we use i-1 and i
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing in daily uptrend
            if bullish_engulf[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing in daily downtrend
            elif bearish_engulf[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish engulfing or trend change
            if bearish_engulf[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish engulfing or trend change
            if bullish_engulf[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h engulfing candles with daily EMA(34) trend filter
# - Bullish/bearish engulfing patterns indicate strong momentum shifts
# - Daily EMA(34) ensures we only trade in the direction of higher timeframe trend
# - Works in bull markets (buy bullish engulfing in uptrend) and bear markets (sell bearish engulfing in downtrend)
# - Engulfing patterns are relatively rare, keeping trade frequency low (target: 15-30/year)
# - Position size 0.25 limits drawdown while allowing meaningful gains
# - Simple, robust logic with clear entry/exit conditions