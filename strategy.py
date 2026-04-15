#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend (ATR=10, mult=3) for trend direction and 1h RSI(14) for mean reversion timing.
# In 12h uptrend (Supertrend green), wait for 1h RSI < 30 to go long (pullback entry).
# In 12h downtrend (Supertrend red), wait for 1h RSI > 70 to go short (bounce entry).
# Volume confirmation (current > 1.5x 20-period SMA) ensures momentum validity.
# Designed for low trade frequency (20-40/year) to minimize fee drag while adapting to trend and mean reversion.
# Works in bull (trend following on pullbacks) and bear (counter-trend bounces in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 12h and 1h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1h = get_htf_data(prices, '1h')
    if len(df_12h) < 30 or len(df_1h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Supertrend (ATR=10, mult=3) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3 * atr_12h)
    lower_band = hl2 - (3 * atr_12h)
    
    # Supertrend
    supertrend = np.full(len(close_12h), np.nan)
    direction = np.full(len(close_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # === 1h Indicators: RSI(14) ===
    close_1h = df_1h['close'].values
    delta = pd.Series(close_1h).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(rsi_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 12h uptrend (Supertrend direction = 1)
        # 2. 1h RSI < 30 (oversold pullback)
        # 3. Volume confirmation
        if (direction_aligned[i] == 1) and (rsi_1h_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 12h downtrend (Supertrend direction = -1)
        # 2. 1h RSI > 70 (overbought bounce)
        # 3. Volume confirmation
        elif (direction_aligned[i] == -1) and (rsi_1h_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Supertrend12h_RSI1h_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0