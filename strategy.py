#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Williams Alligator (13,8,5) for trend direction and 1d RSI(14) for mean reversion timing.
# In 1w uptrend (Alligator jaws < teeth < lips, price > lips), wait for 1d RSI < 30 to go long (pullback entry).
# In 1w downtrend (Alligator jaws > teeth > lips, price < lips), wait for 1d RSI > 70 to go short (bounce entry).
# Volume confirmation ensures momentum validity. Session filter (00-23 UTC) always active for 12h.
# Designed for low trade frequency (12-37/year) to minimize fee drag while adapting to trend and mean reversion.
# Williams Alligator uses SMAs with specific offsets: jaws=13-period SMA shifted 8 bars, teeth=8-period SMA shifted 5 bars, lips=5-period SMA shifted 3 bars.

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
    
    # Get 1w and 1d HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Williams Alligator (13,8,5) ===
    close_1w = df_1w['close'].values
    # Jaws: 13-period SMA shifted by 8 bars
    jaws_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted by 5 bars
    teeth_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted by 3 bars
    lips_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to LTF
    jaws_1w_aligned = align_htf_to_ltf(prices, df_1w, jaws_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
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
        # Williams Alligator conditions
        # Uptrend: jaws < teeth < lips and price > lips
        # Downtrend: jaws > teeth > lips and price < lips
        jaws = jaws_1w_aligned[i]
        teeth = teeth_1w_aligned[i]
        lips = lips_1w_aligned[i]
        price = close[i]
        
        # Check for valid Alligator values (not NaN)
        if np.isnan(jaws) or np.isnan(teeth) or np.isnan(lips):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if RSI is NaN
        if np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1w uptrend (jaws < teeth < lips and price > lips)
        # 2. 1d RSI < 30 (oversold pullback)
        # 3. Volume confirmation
        if (jaws < teeth < lips) and (price > lips) and (rsi_1d_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1w downtrend (jaws > teeth > lips and price < lips)
        # 2. 1d RSI > 70 (overbought bounce)
        # 3. Volume confirmation
        elif (jaws > teeth > lips) and (price < lips) and (rsi_1d_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsAlligator_RSI14_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0