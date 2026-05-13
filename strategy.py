#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeS
Hypothesis: Camarilla R1/S1 breakout with 1-day EMA trend filter and volume confirmation.
Camarilla levels provide institutional support/resistance; EMA50 filters trend direction.
Volume breakout confirms institutional participation. Designed for 20-40 trades/year
to work in both bull and bear markets by following established trends with institutional validation.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeS"
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
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Use previous day's typical price for Camarilla calculation
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = typical_price[0]  # handle first element
    
    # Camarilla R1 and S1 levels
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    r1 = close + (high - low) * 1.1 / 12
    s1 = close - (high - low) * 1.1 / 12
    
    # Calculate RSI(14) for momentum confirmation
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
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with RSI > 50 and volume confirmation
            if close[i] > r1[i] and rsi[i] > 50 and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA50 (uptrend filter)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with RSI < 50 and volume confirmation
            elif close[i] < s1[i] and rsi[i] < 50 and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA50 (downtrend filter)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or RSI < 40
            if close[i] < s1[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or RSI > 60
            if close[i] > r1[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals