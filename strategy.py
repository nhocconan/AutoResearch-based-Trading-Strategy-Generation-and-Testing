#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering. Enter long when price > KAMA, RSI > 50, and CHOP < 61.8 (trending regime). Enter short when price < KAMA, RSI < 50, and CHOP < 61.8. Uses discrete position size 0.25. Designed for 7-25 trades/year on 1d by requiring trending regime (CHOP < 61.8) and alignment of price, KAMA, and RSI, reducing whipsaw in ranging markets while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (optional, but can add confluence)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate KAMA(10, 2, 30) on close
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # Smoothing Constant (SC) = [ER * (fastest - slowest) + slowest]^2
    # where fastest = 2/(2+1), slowest = 2/(30+1)
    # KAMA[i] = KAMA[i-1] + SC * (price[i] - KAMA[i-1])
    
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = close_series.rolling(window=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=False)
    er = change / np.maximum(volatility, 1e-10)
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1) over 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr1 / np.maximum(max_high - min_low, 1e-10)) / np.log10(14)
    
    # Optional: 1w EMA34 for HTF trend filter (confluence)
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (need 10 for ER, but we start from 0), RSI(14), CHOP(14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop[i] < 61.8
        
        # Price vs KAMA for trend direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI for momentum confirmation
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Optional: 1w EMA34 filter for HTF trend alignment
        htf_uptrend = close[i] > ema_34_1w_aligned[i]
        htf_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, trending regime, HTF uptrend (optional)
            long_signal = price_above_kama and rsi_bullish and trending_regime and htf_uptrend
            
            # Short: price < KAMA, RSI < 50, trending regime, HTF downtrend (optional)
            short_signal = price_below_kama and rsi_bearish and trending_regime and htf_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 50 OR regime becomes ranging (CHOP >= 61.8)
            if (price_below_kama or not rsi_bullish or chop[i] >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 50 OR regime becomes ranging (CHOP >= 61.8)
            if (price_above_kama or not rsi_bearish or chop[i] >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0