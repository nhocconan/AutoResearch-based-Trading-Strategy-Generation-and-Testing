#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Filter
Hypothesis: 12-hour KAMA trend direction with RSI momentum filter and volume confirmation.
Trades in direction of KAMA trend (adaptive moving average) only when RSI shows momentum
and volume confirms strength. Works in bull markets by riding trends and in bear markets
by catching sharp reversals when momentum aligns with trend. Targets 15-35 trades/year
by requiring KAMA trend confirmation, RSI extremes, and volume spike.
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
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    close_series = pd.Series(df_1d['close'])
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series - close_series.shift(1)).rolling(window=10, min_periods=10).sum()
    ER = change / volatility.replace(0, np.nan)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    SC = (ER * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = close_series.copy()
    for i in range(1, len(kama)):
        if not np.isnan(SC.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + SC.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = kama.iloc[i-1]
    kama_values = kama.values
    
    # Get weekly data for trend filter (optional but adds robustness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        # If weekly data insufficient, use daily trend only
        ema_20_1w = None
    else:
        ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI calculation on daily closes
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align all higher timeframe data to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    if ema_20_1w is not None:
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_surge = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Additional weekly trend filter if available
        weekly_trend_filter = True
        if ema_20_1w is not None:
            weekly_trend_filter = not np.isnan(ema_20_1w_aligned[i]) and close[i] > ema_20_1w_aligned[i]
        
        # KAMA trend: price above KAMA = bullish, below = bearish
        kama_bullish = close[i] > kama_aligned[i]
        kama_bearish = close[i] < kama_aligned[i]
        
        # RSI conditions: momentum confirmation
        rsi_overbought = rsi_aligned[i] > 60
        rsi_oversold = rsi_aligned[i] < 40
        
        # Entry conditions
        # Long: price above KAMA (bullish trend) + RSI momentum (>60) + volume surge + weekly filter
        long_entry = (kama_bullish and 
                     rsi_overbought and 
                     volume_surge[i] and 
                     weekly_trend_filter)
        
        # Short: price below KAMA (bearish trend) + RSI momentum (<40) + volume surge + weekly filter
        short_entry = (kama_bearish and 
                      rsi_oversold and 
                      volume_surge[i] and 
                      weekly_trend_filter)
        
        # Exit conditions: trend reversal or RSI divergence
        long_exit = (not kama_bullish) or (rsi_aligned[i] < 50)
        short_exit = (not kama_bearish) or (rsi_aligned[i] > 50)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_RSI_Filter"
timeframe = "12h"
leverage = 1.0