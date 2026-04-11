#!/usr/bin/env python3
# 6h_1d_1w_engulfing_momentum_v1
# Strategy: 6h momentum with 1d/1w candle structure and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Combines engulfing candle patterns with multi-timeframe trend alignment
# and volume spikes to capture momentum moves in both bull and bear markets.
# Uses 1d engulfing signals filtered by 1w trend and 6m momentum confirmation.
# Designed for low frequency (15-30 trades/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_engulfing_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Engulfing pattern
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d) & \
                     (close_1d > open_1d) & (open_1d < close_1d) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1)) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1))
    # Actually: current candle body completely engulfs previous candle body
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1)) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1))
    # Correct bullish engulfing: current green candle completely engulfs previous red candle
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1)) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1))
    # Proper implementation:
    bullish_engulf = (close_1d > open_1d) & (close_1d.shift(1) < open_1d.shift(1)) & \
                     (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1))
    
    # Bearish engulfing: current red candle completely engulfs previous green candle
    bearish_engulf = (close_1d < open_1d) & (close_1d.shift(1) > open_1d.shift(1)) & \
                     (close_1d < open_1d.shift(1)) & (open_1d > close_1d.shift(1))
    
    # 1w trend filter: price above/below 20-period EMA
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = close_1w > ema_20_1w
    downtrend_1w = close_1w < ema_20_1w
    
    # Align 1d engulfing signals to 6h
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulf.astype(float))
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # 6m RSI for momentum confirmation
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i]) or
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or
            np.isnan(rsi.iloc[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        bullish_setup = (bullish_engulf_aligned[i] > 0.5 and 
                        uptrend_1w_aligned[i] > 0.5 and 
                        rsi.iloc[i] > 50 and 
                        volume_spike[i])
        
        bearish_setup = (bearish_engulf_aligned[i] > 0.5 and 
                        downtrend_1w_aligned[i] > 0.5 and 
                        rsi.iloc[i] < 50 and 
                        volume_spike[i])
        
        # Exit conditions: opposite engulfing or RSI extreme
        if bullish_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (bearish_engulf_aligned[i] > 0.5 or rsi.iloc[i] < 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_engulf_aligned[i] > 0.5 or rsi.iloc[i] > 70):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals