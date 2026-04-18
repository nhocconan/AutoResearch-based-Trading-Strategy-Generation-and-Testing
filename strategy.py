#!/usr/bin/env python3
"""
4h_4H_Momentum_Regime_v1
Strategy: 4h momentum with 1D trend filter and volatility regime filter.
Long: Price > 20-period high and RSI > 50 in uptrend + low volatility regime.
Short: Price < 20-period low and RSI < 50 in downtrend + low volatility regime.
Volatility regime: ATR(14) < ATR(50) indicates low volatility (trending conditions).
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull/bear via trend filter and volatility regime to avoid whipsaws.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily trend to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h indicators
    # 20-period high/low (Donchian channels)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    
    # Volatility regime: low volatility when short-term ATR < long-term ATR
    vol_regime = atr_14 < atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for EMA200 and ATR50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(rsi[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Momentum conditions
        price_above = close[i] > high_20[i]
        price_below = close[i] < low_20[i]
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        if position == 0:
            # Long: uptrend + price above 20-period high + RSI bullish + low volatility regime
            if uptrend and price_above and rsi_bullish and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price below 20-period low + RSI bearish + low volatility regime
            elif downtrend and price_below and rsi_bearish and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or momentum breakdown
            if not uptrend or close[i] < low_20[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or momentum breakdown
            if not downtrend or close[i] > high_20[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4H_Momentum_Regime_v1"
timeframe = "4h"
leverage = 1.0