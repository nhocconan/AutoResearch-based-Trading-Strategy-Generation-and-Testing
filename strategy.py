#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day KAMA trend with RSI pullback and volume confirmation.
# Long when KAMA turns bullish, price pulls back to KAMA, and volume confirms.
# Short when KAMA turns bearish, price rallies to KAMA, and volume confirms.
# Uses daily KAMA for trend direction, RSI for pullback entries, volume for confirmation.
# Designed to work in trending markets by entering on pullbacks and in ranging markets via mean reversion.
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing.

name = "12h_1dKAMA_RSI_Pullback_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA ( Kaufman Adaptive Moving Average )
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 2)  # EMA(2)
    slow_sc = 2 / (30 + 2)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(df_1d['close'].values))
    volatility = np.sum(np.abs(np.diff(df_1d['close'].values)), axis=0) if len(df_1d) > 1 else 0
    # Vectorized ER calculation
    price_change = np.abs(df_1d['close'].diff().values)
    price_volatility = price_change.rolling(window=er_length, min_periods=1).sum()
    net_change = np.abs(df_1d['close'].diff(er_length).values)
    er = np.where(price_volatility.values != 0, np.abs(net_change) / price_volatility.values, 0)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(df_1d['close'].values, np.nan, dtype=np.float64)
    kama[er_length] = df_1d['close'].iloc[er_length]
    for i in range(er_length + 1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI calculation on 1-day closes
    delta = df_1d['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Volume confirmation: >1.3x 20-period average on 12h timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after KAMA/RSI warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up, price pulls back to KAMA, RSI not overbought
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            price_pullback_to_kama = close[i] <= kama_aligned[i] * 1.005 and close[i] >= kama_aligned[i] * 0.995
            rsi_not_overbought = rsi_aligned[i] < 60
            
            if kama_rising and price_pullback_to_kama and rsi_not_overbought and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, price rallies to KAMA, RSI not oversold
            elif kama_aligned[i] < kama_aligned[i-1] and close[i] >= kama_aligned[i] * 0.995 and close[i] <= kama_aligned[i] * 1.005 and rsi_aligned[i] > 40 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down or RSI overbought
            if kama_aligned[i] < kama_aligned[i-1] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up or RSI oversold
            if kama_aligned[i] > kama_aligned[i-1] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals