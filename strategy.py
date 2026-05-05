#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly KAMA trend with 1d RSI mean reversion and volume spike confirmation
# Long when price < weekly KAMA (bullish bias) AND RSI(14) < 30 AND volume > 1.5 * avg_volume(20) on 1d
# Short when price > weekly KAMA (bearish bias) AND RSI(14) > 70 AND volume > 1.5 * avg_volume(20) on 1d
# Exit when RSI crosses back to neutral (40-60 range) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# Weekly KAMA adapts to trend strength and reduces whipsaw in ranging markets
# 1d RSI(14) provides mean-reversion entries during overextended moves
# Volume spike confirms conviction and reduces false signals
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)

name = "1d_WeeklyKAMA_RSI14_MeanReversion_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for KAMA
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly KAMA ( Kaufman Adaptive Moving Average )
    close_1w_series = pd.Series(close_1w)
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close_1w_series - close_1w_series.shift(10))
    volatility = abs(close_1w_series - close_1w_series.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)  # Avoid division by zero
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    # KAMA calculation
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc.iloc[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Get 1d data ONCE before loop for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for RSI and volume average
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1d_values[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below weekly KAMA (uptrend bias), RSI oversold (<30), volume confirmation, in session
            if close[i] < kama_1w_aligned[i] and rsi_1d_values[i] < 30 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above weekly KAMA (downtrend bias), RSI overbought (>70), volume confirmation, in session
            elif close[i] > kama_1w_aligned[i] and rsi_1d_values[i] > 70 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 40 (mean reversion complete) OR volume drops below average
            if rsi_1d_values[i] > 40 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 60 (mean reversion complete) OR volume drops below average
            if rsi_1d_values[i] < 60 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals