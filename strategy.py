#!/usr/bin/env python3
"""
4h_adaptive_ema_crossover_volatility_regime_v1
Hypothesis: EMA crossover with volatility regime filter (ATR-based) to avoid whipsaws.
In trending markets (high volatility), trade EMA(8)xEMA(21) crossovers.
In ranging markets (low volatility), fade at Bollinger Bands with RSI filter.
Volume confirmation reduces false signals. Designed to work in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adaptive_ema_crossover_volatility_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA crossover components
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR for volatility regime (14-period)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.inf], tr2])  # First value inf to avoid look-ahead
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Bollinger Bands (20, 2) for ranging markets
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    
    # RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: high vol = trending, low vol = ranging
    # Use ATR ratio to ATR moving average
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_ratio = atr / atr_ma  # >1 = high volatility (trending), <1 = low volatility (ranging)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if vol_ratio[i] > 1.2:  # Trending regime (high volatility)
            # EMA crossover signals
            if ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1]:  # Golden cross
                if vol_confirm and close[i] > sma20[i]:  # Additional filter: above SMA20
                    position = 1
                    signals[i] = 0.30
            elif ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1]:  # Death cross
                if vol_confirm and close[i] < sma20[i]:  # Additional filter: below SMA20
                    position = -1
                    signals[i] = -0.30
            # Exit on opposite crossover
            elif position == 1 and ema8[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and ema8[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
        else:  # Ranging regime (low volatility)
            # Mean reversion at Bollinger Bands with RSI filter
            if close[i] <= bb_lower[i] and rsi[i] < 30 and vol_confirm:
                position = 1
                signals[i] = 0.20
            elif close[i] >= bb_upper[i] and rsi[i] > 70 and vol_confirm:
                position = -1
                signals[i] = -0.20
            # Exit when price returns to mean or RSI normalizes
            elif position == 1 and (close[i] >= sma20[i] or rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] <= sma20[i] or rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals