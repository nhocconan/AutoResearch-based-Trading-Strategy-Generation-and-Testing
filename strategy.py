#!/usr/bin/env python3
# 1h_RSI_CCI_4hTrend_1dVolatility
# Hypothesis: Combines RSI and CCI on 1h for mean-reversion entries in the direction of 4h trend,
# filtered by 1d volatility regime. RSI < 30 and CCI < -100 for long, RSI > 70 and CCI > 100 for short.
# Only trades during high volatility periods (ATR(24) > 1.5 * ATR(96)) to avoid ranging markets.
# Works in bull markets via pullbacks in uptrends, and in bear markets via bounces in downtrends.
# Low trade frequency expected due to triple confirmation (RSI, CCI, trend) and volatility filter.

name = "1h_RSI_CCI_4hTrend_1dVolatility"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def cci(high, low, close, period=20):
    """Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=period, min_periods=period).mean()
    mad = pd.Series(typical_price).rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (typical_price - sma_tp.values) / (0.015 * mad.values)

def atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = np.abs(np.subtract(high, low))
    tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
    tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first true range
    return pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d ATR(24) and ATR(96) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_24_1d = atr(high_1d, low_1d, close_1d, 24)
    atr_96_1d = atr(high_1d, low_1d, close_1d, 96)
    atr_24_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_24_1d)
    atr_96_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_96_1d)
    
    # Calculate 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    rsi_1h = rsi(close, 14)
    cci_1h = cci(high, low, close, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA (20) + 1d ATR (96) + 1h RSI/CCI
    start_idx = 100  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_24_1d_aligned[i]) or np.isnan(atr_96_1d_aligned[i]) or
            np.isnan(rsi_1h[i]) or np.isnan(cci_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 4h EMA
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Volatility filter: only trade when ATR(24) > 1.5 * ATR(96)
        high_volatility = atr_24_1d_aligned[i] > 1.5 * atr_96_1d_aligned[i]
        
        if position == 0:
            # Long: RSI oversold, CCI deeply oversold, uptrend, high volatility
            if (rsi_1h[i] < 30 and cci_1h[i] < -100 and uptrend and high_volatility):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought, CCI deeply overbought, downtrend, high volatility
            elif (rsi_1h[i] > 70 and cci_1h[i] > 100 and downtrend and high_volatility):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI overbought OR trend breaks
            if rsi_1h[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI oversold OR trend breaks
            if rsi_1h[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals