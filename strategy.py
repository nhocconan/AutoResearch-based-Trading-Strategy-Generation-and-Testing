#!/usr/bin/env python3
"""
1h Range-Bound Strategy with 4h/1d Trend Filter and Volume Confirmation
In ranges (ADX < 25), fade extremes using RSI with 4h/1d trend filter.
In trends (ADX >= 25), follow 4h/1d trend with pullback entries.
Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_range_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4h Trend Indicators ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # EMA21 and EMA50 for 4h trend
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    # Trend: 1 if EMA21 > EMA50, -1 if EMA21 < EMA50
    trend_4h = np.where(ema21_4h > ema50_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === 1d Trend Filter (stronger) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    trend_1d = np.where(close_1d > sma50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators for Entry Timing ===
    # RSI(14) for mean reversion in ranges
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # ADX(14) for regime detection
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # Regime: ADX < 25 = range, ADX >= 25 = trend
        if adx[i] < 25:
            # RANGE MARKET: Mean reversion at extremes
            if rsi[i] < 30 and trend_1d_aligned[i] == 1 and vol_ratio[i] > 1.2:
                # Oversold in uptrend -> long
                if position != 1:
                    position = 1
                    signals[i] = 0.20
                else:
                    signals[i] = 0.20
            elif rsi[i] > 70 and trend_1d_aligned[i] == -1 and vol_ratio[i] > 1.2:
                # Overbought in downtrend -> short
                if position != -1:
                    position = -1
                    signals[i] = -0.20
                else:
                    signals[i] = -0.20
            else:
                # No clear signal, reduce to zero if not already flat
                if position != 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0
        else:
            # TREND MARKET: Follow 4h/1d trend on pullbacks
            if trend_4h_aligned[i] == 1 and trend_1d_aligned[i] == 1:
                # Uptrend: buy pullbacks
                if rsi[i] < 40 and vol_ratio[i] > 1.1:
                    if position != 1:
                        position = 1
                        signals[i] = 0.20
                    else:
                        signals[i] = 0.20
                elif rsi[i] > 60:
                    # Exit long on overbought
                    if position != 0:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.0
                else:
                    # Hold trend position
                    if position == 1:
                        signals[i] = 0.20
                    else:
                        signals[i] = 0.0
            elif trend_4h_aligned[i] == -1 and trend_1d_aligned[i] == -1:
                # Downtrend: sell rallies
                if rsi[i] > 60 and vol_ratio[i] > 1.1:
                    if position != -1:
                        position = -1
                        signals[i] = -0.20
                    else:
                        signals[i] = -0.20
                elif rsi[i] < 40:
                    # Exit short on oversold
                    if position != 0:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.0
                else:
                    # Hold trend position
                    if position == -1:
                        signals[i] = -0.20
                    else:
                        signals[i] = 0.0
            else:
                # Mixed trends or no clear trend -> flat
                if position != 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0
    
    return signals