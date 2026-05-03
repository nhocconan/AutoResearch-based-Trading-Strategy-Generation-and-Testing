#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based volatility filter.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (high < EMA13) in bull trend (close > 1d EMA50) with ATR(14) > 0.5x ATR(50) (ensuring sufficient volatility).
# Short when Bear Power < 0 (high < EMA13) AND Bull Power < 0 (low > EMA13) in bear trend (close < 1d EMA50) with ATR(14) > 0.5x ATR(50).
# Uses discrete position sizing (0.25) to minimize fee churn. The Elder Ray identifies institutional buying/selling pressure via EMA13.
# The 1d EMA50 provides higher timeframe trend filter to avoid counter-trend trades. ATR filter ensures trades occur in sufficient volatility regimes.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "6h_ElderRay_1dEMA50_ATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13  # Bull Power: close - EMA13
    bear_power = high - ema_13   # Bear Power: high - EMA13 (note: typically high - EMA, but we'll adjust logic)
    
    # ATR for volatility filter (14 and 50 periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    volatility_filter = atr_ratio > 0.5  # Require short-term volatility > 50% of long-term
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_filter = volatility_filter[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Elder Ray conditions: Bull Power > 0 AND Bear Power < 0 for long setup
        # Bear Power < 0 AND Bull Power < 0 for short setup (both below EMA13)
        long_setup = bp > 0 and br < 0
        short_setup = br < 0 and bp < 0  # Both negative = bearish momentum
        
        # Entry logic
        if position == 0:
            if is_bull_trend and long_setup and vol_filter:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and short_setup and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power > 0 (bullish failure) OR trend reversal
            if br > 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 (bearish failure) OR trend reversal
            if bp > 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals