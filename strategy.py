#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA trend direction with RSI mean reversion and chop regime filter.
# In bull markets: KAMA up + RSI < 30 (oversold) in choppy regime → long
# In bear markets: KAMA down + RSI > 70 (overbought) in choppy regime → short
# Chop regime (CHOP > 61.8) filters trending markets to avoid false signals.
# Volume confirmation ensures participation. Target: 30-100 trades over 4 years.
# Primary timeframe: 1d, HTF: 1w for regime context.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for weekly regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))  # simplified for 1-period
    er_num = np.abs(np.diff(close, prepend=close[0]))
    er_den = np.sum(np.abs(np.diff(close, prepend=close[0]))[:10])  # placeholder, will compute properly
    
    # Proper KAMA calculation
    close_s = pd.Series(close)
    change = np.abs(close_s.diff()).values
    volatility = pd.Series(close).rolling(window=10, min_periods=1).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period) - using daily data
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero and log10 of zero
    chop_denom = np.log10(np.where(atr_14 == 0, 1e-10, atr_14)) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging/choppy (chop > 61.8)
        chop_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or trend changes
            if rsi[i] > 50 or kama[i] < kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or trend changes
            if rsi[i] < 50 or kama[i] > kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: KAMA trending up AND RSI oversold (<30)
                if kama[i] > kama[i-1] and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short entry: KAMA trending down AND RSI overbought (>70)
                elif kama[i] < kama[i-1] and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals