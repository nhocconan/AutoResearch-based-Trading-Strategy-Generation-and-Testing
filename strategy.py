#!/usr/bin/env python3
"""
6h_12h_ADX_Trend_Reversal
Hypothesis: In 12h trends (ADX > 25), 6h reversals occur when price rejects 12h EMA21 with RSI divergence.
Works in bull/bear by trading pullbacks in strong trends. Uses ADX trend filter + RSI divergence for precise entries.
Target: 20-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr = np.maximum(np.absolute(np.diff(high, prepend=high[0])),
                    np.maximum(np.absolute(np.diff(low, prepend=low[0])),
                               np.absolute(np.diff(close, prepend=close[0]))))
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
    
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX trend filter
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    strong_trend = adx_12h > 25
    
    # 12h EMA21 for dynamic support/resistance
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 6h RSI for divergence detection
    rsi_6h = calculate_rsi(close, 14)
    
    # Align 12h indicators to 6h
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend.astype(float))
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(strong_trend_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or
            np.isnan(rsi_6h[i])):
            signals[i] = 0.0
            continue
        
        # Bullish reversal: price rejects 12h EMA21 from below with RSI oversold
        bullish_reject = (close[i] > ema_21_12h_aligned[i] and 
                         low[i] <= ema_21_12h_aligned[i] * 1.002 and  # within 0.2% of EMA
                         rsi_6h[i] < 35 and
                         strong_trend_aligned[i] > 0.5)
        
        # Bearish reversal: price rejects 12h EMA21 from above with RSI overbought
        bearish_reject = (close[i] < ema_21_12h_aligned[i] and 
                         high[i] >= ema_21_12h_aligned[i] * 0.998 and  # within 0.2% of EMA
                         rsi_6h[i] > 65 and
                         strong_trend_aligned[i] > 0.5)
        
        if bullish_reject and position != 1:
            position = 1
            signals[i] = position_size
        elif bearish_reject and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Exit on trend weakening or opposite signal
            if position == 1 and (strong_trend_aligned[i] <= 0.5 or bearish_reject):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (strong_trend_aligned[i] <= 0.5 or bullish_reject):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_12h_ADX_Trend_Reversal"
timeframe = "6h"
leverage = 1.0