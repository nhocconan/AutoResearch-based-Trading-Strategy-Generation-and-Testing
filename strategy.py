#!/usr/bin/env python3
"""
1h_4h_1d_Combined_Momentum_Trend_v1
Hypothesis: Combine 1h momentum with 4h trend and 1d regime filter for high-conviction trades.
- 4h EMA200 determines trend direction (bull/bear)
- 1d ADX > 25 filters for trending markets only
- 1h RSI(14) pulls back to 40-60 in trend direction for entry
- Volume confirmation: 1.5x average volume
- Target: 15-30 trades/year by requiring multiple timeframe alignment
- Works in bull (trend continuation) and bear (trend continuation on rebounds)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Combined_Momentum_Trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H TREND: EMA200 ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1D REGIME: ADX > 25 for trending markets ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(abs(high_1d[1:] - high_1d[:-1]), 
                               abs(low_1d[1:] - low_1d[:-1])))
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 13:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[max(0, i-12):i+1]) if i >= 12 else np.mean(tr[:i+1])
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1H MOMENTUM: RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h trend
        uptrend_4h = close[i] > ema200_4h_aligned[i]
        downtrend_4h = close[i] < ema200_4h_aligned[i]
        
        # 1d regime: trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # RSI conditions for entry
        rsi_bullish = 40 <= rsi[i] <= 60  # Pullback in uptrend
        rsi_bearish = 40 <= rsi[i] <= 60  # Pullback in downtrend
        
        # Entry logic
        long_entry = (uptrend_4h and trending and rsi_bullish and strong_volume)
        short_entry = (downtrend_4h and trending and rsi_bearish and strong_volume)
        
        # Exit: trend reversal or loss of momentum
        exit_long = (position == 1 and (not uptrend_4h or not trending))
        exit_short = (position == -1 and (not downtrend_4h or not trending))
        
        # Execute trades
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals