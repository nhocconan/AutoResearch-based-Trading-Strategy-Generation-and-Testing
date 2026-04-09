#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA trend direction + RSI extremes + Choppiness regime filter.
# In bull markets: KAMA up + RSI<30 (oversold) in choppy regime → long
# In bear markets: KAMA down + RSI>70 (overbought) in choppy regime → short
# Chop regime (CHOP>61.8) avoids trending whipsaws; mean reversion works best in ranging markets.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 10-25 trades/year.

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
    
    # 1w HTF for trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) - 1d
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(10))
    volatility = close_s.diff().abs().rolling(10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0.01)  # fallback when volatility=0
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # RSI(14) - 1d
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when no loss
    
    # Choppiness Index (14) - 1d
    # True Range
    tr1 = high - low
    tr2 = abs(high - close_s.shift(1))
    tr3 = abs(low - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).sum()
    # Highest high and lowest low over 14 periods
    hh14 = high.rolling(14, min_periods=14).max()
    ll14 = low.rolling(14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14)
    chop = chop.fillna(50)  # neutral when range=0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) or chop regime ends (trending)
            if rsi[i] > 60 or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) or chop regime ends (trending)
            if rsi[i] < 40 or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop[i] > 61.8:  # Chop regime: ranging market
                # Long: KAMA up + RSI oversold (<30)
                if close[i] > kama[i] and rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short: KAMA down + RSI overbought (>70)
                elif close[i] < kama[i] and rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals