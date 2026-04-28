#!/usr/bin/env python3
"""
4h_Keltner_Channel_MeanReversion
Hypothesis: Price tends to revert to the mean after touching Keltner Channel bands during low volatility periods.
Long when price touches lower band with bullish divergence on RSI and low volatility (ATR ratio < 0.8).
Short when price touches upper band with bearish divergence on RSI and low volatility.
Uses 1-day EMA200 trend filter to avoid counter-trend trades in strong trends.
Designed for 20-30 trades/year to minimize fee drag while capturing mean reversion in ranging markets.
Works in both bull and bear by adapting to volatility regimes and using trend filter for direction bias.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate ATR(14) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate EMA20 for Keltner Channel middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands (typically 1.5x ATR)
    kc_upper = ema20 + (1.5 * atr)
    kc_lower = ema20 - (1.5 * atr)
    
    # Calculate RSI(14) for divergence
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR ratio for volatility filter (current ATR / 50-period average ATR)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma_50 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or
            np.isnan(rsi[i]) or
            np.isnan(rsi[i-1]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Price touching Keltner bands
        touch_lower = low[i] <= kc_lower[i]
        touch_upper = high[i] >= kc_upper[i]
        
        # RSI divergence signals
        # Bullish RSI divergence: price makes lower low, RSI makes higher low
        bullish_div = (low[i] < low[i-1]) and (rsi[i] > rsi[i-1])
        # Bearish RSI divergence: price makes higher high, RSI makes lower high
        bearish_div = (high[i] > high[i-1]) and (rsi[i] < rsi[i-1])
        
        # Volatility filter: low volatility environment (ATR ratio < 0.8)
        low_vol = atr_ratio[i] < 0.8
        
        # Trend filter from 1d EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        long_entry = touch_lower and bullish_div and low_vol and uptrend
        short_entry = touch_upper and bearish_div and low_vol and downtrend
        
        # Exit when price returns to middle line (EMA20)
        long_exit = close[i] >= ema20[i]
        short_exit = close[i] <= ema20[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Keltner_Channel_MeanReversion"
timeframe = "4h"
leverage = 1.0