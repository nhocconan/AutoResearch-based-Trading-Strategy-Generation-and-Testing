#!/usr/bin/env python3
"""
4h_Keltner_Channel_RSI_Momentum
Hypothesis: Keltner Channel breakouts with RSI momentum capture trends in both bull and bear markets.
The middle EMA acts as dynamic support/resistance, while RSI filters momentum strength.
Designed for low trade frequency (20-40/year) with trend-following logic that adapts to market regimes.
"""

name = "4h_Keltner_Channel_RSI_Momentum"
timeframe = "4h"
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
    
    # Calculate Keltner Channel (20, 2.0)
    ema_middle = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_channel = ema_middle + 2.0 * atr
    lower_channel = ema_middle - 2.0 * atr
    
    # Calculate RSI(14)
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
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above upper channel with RSI > 50 and volume confirmation
            if close[i] > upper_channel[i] and rsi[i] > 50 and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA50 (uptrend filter)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below lower channel with RSI < 50 and volume confirmation
            elif close[i] < lower_channel[i] and rsi[i] < 50 and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA50 (downtrend filter)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below middle EMA (trend change) or RSI < 40
            if close[i] < ema_middle[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above middle EMA (trend change) or RSI > 60
            if close[i] > ema_middle[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals