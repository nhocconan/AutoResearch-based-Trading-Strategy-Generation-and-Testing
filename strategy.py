#3911786415010975712
#!/usr/bin/env python3
"""
6h_Adaptive_Trend_Momentum
Hypothesis: Combines adaptive trend detection (ADX) with momentum confirmation (RSI) and volume filters.
Uses 1-day timeframe for trend context and 6h for entry timing. Designed to work in both bull and bear
markets by filtering trades with strong trend strength (ADX > 25) and avoiding choppy markets.
Target: 20-50 trades/year per symbol with disciplined risk control via trend-based exits.
"""

name = "6h_Adaptive_Trend_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Get 1-day trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Warmup for ADX/RSI
        if position == 0:
            # LONG: Strong uptrend (ADX > 25, +DI > -DI) + RSI momentum (> 50) + volume
            if (adx[i] > 25 and plus_di[i] > minus_di[i] and 
                rsi[i] > 50 and volume_confirm[i]):
                # Additional filter: price above 1-day EMA50 (uptrend confirmation)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Strong downtrend (ADX > 25, -DI > +DI) + RSI weakness (< 50) + volume
            elif (adx[i] > 25 and minus_di[i] > plus_di[i] and 
                  rsi[i] < 50 and volume_confirm[i]):
                # Additional filter: price below 1-day EMA50 (downtrend confirmation)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening (ADX < 20) or RSI overextended (< 30) or trend reversal
            if (adx[i] < 20 or rsi[i] < 30 or 
                minus_di[i] > plus_di[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakening (ADX < 20) or RSI overextended (> 70) or trend reversal
            if (adx[i] < 20 or rsi[i] > 70 or 
                plus_di[i] > minus_di[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals