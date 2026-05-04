#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume spike
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on 1d timeframe
# RSI(2) for short-term mean reversion entries (extreme readings)
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Designed to work in both bull and bear markets by following 1d trend and fading extremes
# Target: 20-60 trades/year (80-240 total over 4 years) to balance opportunity and cost
# Prioritizes BTC/ETH performance

name = "1d_KAMA_Trend_RSI2_MeanRev_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe context (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    close_series = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) >= 11 else np.full(len(close), np.nan)
    # Calculate volatility as sum of absolute changes over 10 periods
    volatility = pd.Series(close).rolling(10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1w EMA20 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate RSI(2) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period EMA on volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Determine higher timeframe trend (1w EMA20 direction)
        # For simplicity, use price above/below EMA as trend filter
        htf_uptrend = close[i] > ema_20_1w_aligned[i]
        htf_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI(2) extreme readings for mean reversion
        rsi_oversold = rsi[i] < 10   # Extreme oversold
        rsi_overbought = rsi[i] > 90  # Extreme overbought
        
        if position == 0:
            # Long: Extreme oversold + volume spike + price above KAMA (uptrend alignment) + HTF uptrend
            if (rsi_oversold and volume_spike and price_above_kama and htf_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Extreme overbought + volume spike + price below KAMA (downtrend alignment) + HTF downtrend
            elif (rsi_overbought and volume_spike and price_below_kama and htf_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (50) OR price below KAMA (trend change)
            if rsi[i] >= 50 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (50) OR price above KAMA (trend change)
            if rsi[i] <= 50 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals