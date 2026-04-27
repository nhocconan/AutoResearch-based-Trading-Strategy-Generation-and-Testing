# 1d_Weekly_EMA_Crossover_Trend_Filtered
# Hypothesis: Weekly EMA crossover provides strong directional bias, filtered by daily momentum and volume to avoid whipsaws.
# Weekly EMA crossover (21/55) determines trend direction. Daily RSI filters entries to avoid overextended moves.
# Volume spike confirms breakout strength. Position size 0.25 for risk control.
# Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trends.
# Works in bull markets via trend continuation and bear markets via trend reversals.

#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (EMA21/55 crossover)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55_1w = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    ema_55_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_55_1w)
    
    # Get daily data for momentum filter (RSI14) and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate daily RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (55), daily RSI (14), volume MA (20)
    start_idx = max(55, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_55_1w_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_fast = ema_21_1w_aligned[i]
        ema_slow = ema_55_1w_aligned[i]
        rsi = rsi_14_1d_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Trend: weekly EMA crossover
        bullish_trend = ema_fast > ema_slow
        bearish_trend = ema_fast < ema_slow
        
        # Momentum filter: RSI not extreme (avoid chasing)
        mom_filter = (rsi > 30) and (rsi < 70)
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: bullish weekly trend + RSI not overbought + volume spike
            if bullish_trend and mom_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: bearish weekly trend + RSI not oversold + volume spike
            elif bearish_trend and mom_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: weekly trend turns bearish or RSI overbought
            if not bullish_trend or rsi >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: weekly trend turns bullish or RSI oversold
            if not bearish_trend or rsi <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_EMA_Crossover_Trend_Filtered"
timeframe = "1d"
leverage = 1.0