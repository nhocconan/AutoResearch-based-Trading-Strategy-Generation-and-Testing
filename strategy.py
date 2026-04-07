#!/usr/bin/env python3
"""
4h_rsi_pullback_1d_trend_volume_v2
Hypothesis: RSI pullback entries are too frequent. Improve by requiring:
1. Stronger trend filter (1d EMA100 instead of EMA50) to avoid chop
2. Volume > 2.0x average (stricter) to ensure institutional participation
3. RSI range narrowed: long 35-45, short 55-65 to capture deeper pullbacks
4. Added ADX(14) > 20 filter to ensure trending environment
Reduces trade frequency while maintaining edge in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Calculate 1d EMA100 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, min_periods=100, adjust=False).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Filters: volume > 2.0x average AND ADX > 20 (trending market)
        vol_ok = volume[i] > (vol_ma[i] * 2.0)
        trend_ok = adx[i] > 20
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 35 (end of pullback) or trend changes
            if rsi[i] < 35 or close[i] < ema_100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 65 (end of pullback) or trend changes
            if rsi[i] > 65 or close[i] > ema_100_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and trend_ok:
                # Long: RSI pullback to 35-45 in uptrend (price above 1d EMA100)
                if (35 <= rsi[i] <= 45 and 
                    close[i] > ema_100_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: RSI pullback to 55-65 in downtrend (price below 1d EMA100)
                elif (55 <= rsi[i] <= 65 and 
                      close[i] < ema_100_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals