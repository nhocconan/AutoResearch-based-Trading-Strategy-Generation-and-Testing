#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_With_Volume_Filter_v1
Hypothesis: Use KAMA (adaptive trend) for direction, RSI for momentum confirmation, and volume spike for institutional participation. KAMA adapts to market noise, reducing false signals in chop. RSI filters extreme momentum. Volume ensures breakouts have follow-through. Designed for low trade frequency (<30/year) to minimize fee drag while capturing trends in both bull and bear markets.
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
    
    # KAMA (adaptive trend) - 10 period
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [0] * len(close)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # RSI (14)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need KAMA, RSI, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        ema_1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50 (bullish momentum), volume spike, above 1d EMA
            if price > kama_val and rsi_val > 50 and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), volume spike, below 1d EMA
            elif price < kama_val and rsi_val < 50 and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or RSI < 40 (loss of momentum)
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or RSI > 60 (loss of bearish momentum)
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_RSI_Trend_With_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0