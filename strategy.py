#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d EMA200 Trend Filter and Volume Confirmation.
- Bollinger Band squeeze (low volatility) precedes explosive moves; breakout captures trend initiation.
- 1d EMA200 provides higher-timeframe trend filter to align with long-term momentum and avoid counter-trend trades.
- Volume confirmation ensures breakout is supported by participation, reducing false signals.
- Position size 0.25 balances profit potential and drawdown control in choppy 6h markets.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag on 6h timeframe.
- Works in bull/bear markets via 1d trend filter and volatility-based logic (squeeze exploits low volatility regimes).
"""

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
    
    # Get 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = (upper - lower) / sma  # normalized bandwidth
    
    # Bollinger Squeeze: bandwidth below 20-period average bandwidth (low volatility regime)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20, 200) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade during squeeze breakout with volume confirmation and trend alignment
            if squeeze[i-1] and not squeeze[i]:  # squeeze breakout (bandwidth expanding)
                if volume_confirm[i]:
                    # Long: break above upper band + above 1d EMA200 (bullish higher-timeframe trend)
                    if close[i] > upper[i] and close[i] > ema_200_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: break below lower band + below 1d EMA200 (bearish higher-timeframe trend)
                    elif close[i] < lower[i] and close[i] < ema_200_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below middle band (SMA) OR below 1d EMA200 (trend change)
            if close[i] < sma[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle band (SMA) OR above 1d EMA200 (trend change)
            if close[i] > sma[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_1dEMA200_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0