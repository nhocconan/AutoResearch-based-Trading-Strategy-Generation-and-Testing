#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with weekly RSI and volume confirmation
# KAMA adapts to market noise - follows price in trends, stays flat in sideways
# Weekly RSI filter ensures alignment with higher timeframe momentum
# Volume surge confirms institutional participation in the move
# Designed for 1d timeframe with low trade frequency (15-30/year) to minimize fee drag
# Works in bull markets (trend following) and bear markets (mean reversion via RSI extremes)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for RSI filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI (14) for momentum filter
    close_1w = df_1w['close'].values
    delta = pd.Series(close_1w).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate KAMA (adaptive moving average) on daily data
    # Efficiency Ratio (ER) over 10 periods
    price_change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[10:] = price_change[9:] / volatility[9:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with close
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 2.0x average volume (50-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for volume average and KAMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price above/below KAMA
        above_kama = price > kama[i]
        
        if position == 0:
            # Long: price crosses above KAMA with RSI > 50 and volume surge
            if price > kama[i] and rsi_1w_aligned[i] > 50 and vol > 2.0 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price crosses below KAMA with RSI < 50 and volume surge
            elif price < kama[i] and rsi_1w_aligned[i] < 50 and vol > 2.0 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI < 40
            if price < kama[i] or rsi_1w_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI > 60
            if price > kama[i] or rsi_1w_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_WeeklyRSI_Volume"
timeframe = "1d"
leverage = 1.0