#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA(34) for trend direction and 1d RSI(14) for mean reversion timing.
# In 4h uptrend (price > EMA34), wait for 1d RSI < 30 to go long (pullback entry).
# In 4h downtrend (price < EMA34), wait for 1d RSI > 70 to go short (bounce entry).
# Volume confirmation ensures momentum validity. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (15-30/year) to minimize fee drag while adapting to trend and mean reversion.

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
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 35 or len(df_1d) < 15:
        return np.zeros(n)
    
    # === 4h Indicators: EMA(34) ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d Indicators: RSI(14) ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
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
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 4h uptrend (price > 4h EMA34)
        # 2. 1d RSI < 30 (oversold pullback)
        # 3. Volume confirmation
        if (close[i] > ema_34_4h_aligned[i]) and (rsi_1d_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. In 4h downtrend (price < 4h EMA34)
        # 2. 1d RSI > 70 (overbought bounce)
        # 3. Volume confirmation
        elif (close[i] < ema_34_4h_aligned[i]) and (rsi_1d_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA34_RSI14_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0