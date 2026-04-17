#!/usr/bin/env python3
"""
4h_Momentum_Divergence_Scalper_v1
Hypothesis: In BTC/ETH, momentum divergences on 4h (bullish: price makes lower low, RSI makes higher low) signal trend reversals in choppy/low-volume environments. Combined with volume confirmation and ADX trend filter, this captures mean-reversion bounces in bear markets and pullbacks in bull markets, with low trade frequency and controlled drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ADX(14) for trend strength filter
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Swing low/high detection for divergence (lookback 5 bars)
    def find_swing_low(arr, lookback=5):
        lows = np.zeros_like(arr)
        lows[:] = np.nan
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.min(arr[i-lookback:i+lookback+1]):
                lows[i] = arr[i]
        return lows
    
    def find_swing_high(arr, lookback=5):
        highs = np.zeros_like(arr)
        highs[:] = np.nan
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.max(arr[i-lookback:i+lookback+1]):
                highs[i] = arr[i]
        return highs
    
    price_swing_low = find_swing_low(low, 5)
    price_swing_high = find_swing_high(high, 5)
    rsi_swing_low = find_swing_low(rsi, 5)
    rsi_swing_high = find_swing_high(rsi, 5)
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    last_price_low = np.nan
    last_rsi_low = np.nan
    last_price_high = np.nan
    last_rsi_high = np.nan
    
    for i in range(n):
        if not np.isnan(price_swing_low[i]) and not np.isnan(rsi_swing_low[i]):
            if np.isnan(last_price_low) or (low[i] < last_price_low and rsi[i] > last_rsi_low):
                bull_div[i] = True
                last_price_low = low[i]
                last_rsi_low = rsi[i]
            else:
                last_price_low = low[i]
                last_rsi_low = rsi[i]
        elif not np.isnan(price_swing_low[i]):
            last_price_low = low[i]
            last_rsi_low = rsi[i]
        
        if not np.isnan(price_swing_high[i]) and not np.isnan(rsi_swing_high[i]):
            if np.isnan(last_price_high) or (high[i] > last_price_high and rsi[i] < last_rsi_high):
                bear_div[i] = True
                last_price_high = high[i]
                last_rsi_high = rsi[i]
            else:
                last_price_high = high[i]
                last_rsi_high = rsi[i]
        elif not np.isnan(price_swing_high[i]):
            last_price_high = high[i]
            last_rsi_high = rsi[i]
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # RSI, ADX, vol lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        vol_filter = volume[i] > (1.3 * vol_ma20[i])
        
        # ADX filter: only trade when trend is weak (ADX < 25) to avoid strong trends
        adx_filter = adx[i] < 25
        
        if position == 0:
            # Long: bullish divergence + volume + low ADX
            if bull_div[i] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + volume + low ADX
            elif bear_div[i] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish divergence OR RSI > 70 (overbought)
            if bear_div[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish divergence OR RSI < 30 (oversold)
            if bull_div[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Momentum_Divergence_Scalper_v1"
timeframe = "4h"
leverage = 1.0