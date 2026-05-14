#!/usr/bin/env python3
# Hypothesis: 1h KAMA trend filter with 4h RSI mean reversion and volume confirmation.
# Long when 1h price > KAMA(ER=10, FAST=2, SLOW=30) AND 4h RSI(14) < 30 AND 1h volume > 1.5x 20-period average.
# Short when 1h price < KAMA(ER=10, FAST=2, SLOW=30) AND 4h RSI(14) > 70 AND 1h volume > 1.5x 20-period average.
# Exit when price crosses KAMA in opposite direction.
# Session filter: 08-20 UTC to avoid low-liquidity periods.
# Uses discrete position sizing (0.20) to minimize fee churn.
# Works in bull/bear: KAMA adapts to trend strength and volatility, 4h RSI identifies overextended conditions for mean reversion within the trend, volume confirms participation.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_KAMA_Trend_4hRSI_MeanReversion_Volume_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # 1h KAMA (ER=10, FAST=2, SLOW=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).clip(0, 1)
    sc = (er * (2/2 - 30/30) + 30/30) ** 2  # FAST=2, SLOW=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1h = volume > (1.5 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h RSI(14)
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)  # Fill NaN with neutral 50
    
    # 4h RSI oversold/overbought
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Align 4h indicators to 1h
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_4h, rsi_oversold.astype(float))
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_4h, rsi_overbought.astype(float))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_oversold_aligned[i]) or
            np.isnan(rsi_overbought_aligned[i]) or
            np.isnan(volume_confirm_1h[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > KAMA + 4h RSI oversold + 1h volume confirmation
            if (close[i] > kama[i] and 
                rsi_oversold_aligned[i] > 0.5 and 
                volume_confirm_1h[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price < KAMA + 4h RSI overbought + 1h volume confirmation
            elif (close[i] < kama[i] and 
                  rsi_overbought_aligned[i] > 0.5 and
                  volume_confirm_1h[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals