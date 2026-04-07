# Solution: 1h RSI(2) Mean Reversion with 4h/1d Trend Filter and Session Filter
# Hypothesis: In ranging markets (common in 2025), RSI(2) identifies short-term extremes for mean reversion.
# Uses 4h EMA50 and 1d EMA200 for trend alignment. Only trades during active hours (08-20 UTC).
# Small position size (0.20) and strict filters target ~15-35 trades/year to avoid fee drag.
# Works in bull (trend-following bias) and bear (mean reversion in ranges).

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi2_meanrev_4h1d_trend_session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === RSI(2) for mean reversion signals ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h trend filter (EMA 50) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d trend filter (EMA 200) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(hours[i])):
            signals[i] = 0.0
            continue
        
        # Check session: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                # Exit positions outside session
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # RSI extreme + trend alignment (both timeframes agree)
            if rsi[i] < 10 and close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]:
                # Oversold + uptrend on both 4h and 1d -> long
                position = 1
                signals[i] = 0.20
            elif rsi[i] > 90 and close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]:
                # Overbought + downtrend on both 4h and 1d -> short
                position = -1
                signals[i] = -0.20
    
    return signals