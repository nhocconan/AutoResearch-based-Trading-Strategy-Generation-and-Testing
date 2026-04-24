#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend direction + RSI mean reversion + choppiness regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for regime context (choppiness index).
- Entry: Long when KAMA turns up AND RSI < 40 AND market is choppy (CHOP > 61.8).
         Short when KAMA turns down AND RSI > 60 AND market is choppy (CHOP > 61.8).
- Exit: Opposite KAMA turn OR RSI crosses 50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- KAMA adapts to market noise, reducing whipsaw in ranging markets.
- RSI extremes in choppy regimes provide high-probability mean reversion entries.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
- Estimated trades: ~60 total over 4 years (~15/year) based on regime-filtered signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close = pd.Series(close)
    direction = np.abs(close.diff(er_period))
    volatility = close.diff().abs().rolling(er_period, min_periods=1).sum()
    er = direction / (volatility + 1e-10)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1))**2
    kama = np.zeros_like(close)
    kama[0] = close.iloc[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1])
    return kama

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close = pd.Series(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    atr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_sum = atr.rolling(period, min_periods=period).sum()
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA trend
    kama_vals = kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Calculate RSI
    rsi_vals = rsi(close, period=14)
    
    # Calculate 1w choppiness index for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    chop_1w = choppiness_index(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values,
        period=14
    )
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions: opposite KAMA turn OR RSI crosses 50
        if position != 0:
            kama_up = kama_vals[i] > kama_vals[i-1]
            kama_down = kama_vals[i] < kama_vals[i-1]
            rsi_above_50 = rsi_vals[i] > 50
            rsi_below_50 = rsi_vals[i] < 50
            
            if position == 1:
                # Exit long: KAMA turns down OR RSI crosses below 50
                if not kama_up or rsi_below_50:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                # Exit short: KAMA turns up OR RSI crosses above 50
                if not kama_down or rsi_above_50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: KAMA turn + RSI extreme + choppy regime
        if position == 0:
            kama_up = kama_vals[i] > kama_vals[i-1]
            kama_down = kama_vals[i] < kama_vals[i-1]
            rsi_oversold = rsi_vals[i] < 40
            rsi_overbought = rsi_vals[i] > 60
            choppy = chop_1w_aligned[i] > 61.8  # Choppy regime threshold
            
            if kama_up and rsi_oversold and choppy:
                signals[i] = 0.25
                position = 1
            elif kama_down and rsi_overbought and choppy:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0