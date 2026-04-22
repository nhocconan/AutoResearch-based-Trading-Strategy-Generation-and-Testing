#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA + RSI + chop filter for mean reversion
    # KAMA adapts to market noise: tracks price in trending, stays flat in chop
    # RSI identifies overbought/oversold extremes
    # Chop filter avoids whipsaw in ranging markets
    # Target: 10-25 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio = |change| / sum(|changes|)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.where(abs_change > 0, change / abs_change, 0)
    # Smooth ER with smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop Index(14) on 1d for regime filter
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                               np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    
    # Align indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike filter (20-period on 1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below KAMA (oversold) + RSI < 30 + Chop > 61.8 (ranging) + volume spike
            if close[i] < kama_aligned[i] and rsi_aligned[i] < 30 and chop_aligned[i] > 61.8 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above KAMA (overbought) + RSI > 70 + Chop > 61.8 (ranging) + volume spike
            elif close[i] > kama_aligned[i] and rsi_aligned[i] > 70 and chop_aligned[i] > 61.8 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses KAMA or RSI returns to neutral
            if position == 1:
                if close[i] > kama_aligned[i] or rsi_aligned[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] < kama_aligned[i] or rsi_aligned[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Volume_Session_v1"
timeframe = "1d"
leverage = 1.0