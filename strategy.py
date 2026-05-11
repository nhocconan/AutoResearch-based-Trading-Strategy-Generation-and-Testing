#!/usr/bin/env python3
# 12h_1d_RSI_Divergence_Confluence_Trend
# Hypothesis: Combining RSI divergence on 1d with trend following on 12h creates high-probability entries.
# In bull markets: bullish RSI divergence (higher low in RSI, lower low in price) + 12h price above EMA50 = long.
# In bear markets: bearish RSI divergence (lower high in RSI, higher high in price) + 12h price below EMA50 = short.
# Volume confirmation filters low-momentum signals. Designed for 20-30 trades/year to minimize fee drag.

name = "12h_1d_RSI_Divergence_Confluence_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get multi-timeframe data
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d RSI(14) for divergence detection ---
    rsi_period = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align 1d RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # --- 12h EMA50 for trend filter ---
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume confirmation (2.0x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RSI calculation
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_aligned[i]) or
            np.isnan(ema_50[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Need at least 3 points for divergence check
        if i < start_idx + 2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish RSI divergence: price makes lower low, RSI makes higher low
            # Check last 3 points for swing low
            if (low[i-2] > low[i] and 
                low[i-1] > low[i] and
                rsi_aligned[i-2] < rsi_aligned[i] and
                rsi_aligned[i-1] < rsi_aligned[i] and
                ema_50[i] < close[i] and  # 12h uptrend
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Bearish RSI divergence: price makes higher high, RSI makes lower high
            elif (high[i-2] < high[i] and 
                  high[i-1] < high[i] and
                  rsi_aligned[i-2] > rsi_aligned[i] and
                  rsi_aligned[i-1] > rsi_aligned[i] and
                  ema_50[i] > close[i] and  # 12h downtrend
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks below EMA50 OR RSI shows bearish divergence
                if (close[i] < ema_50[i] or
                    (high[i-2] < high[i] and 
                     high[i-1] < high[i] and
                     rsi_aligned[i-2] > rsi_aligned[i] and
                     rsi_aligned[i-1] > rsi_aligned[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above EMA50 OR RSI shows bullish divergence
                if (close[i] > ema_50[i] or
                    (low[i-2] > low[i] and 
                     low[i-1] > low[i] and
                     rsi_aligned[i-2] < rsi_aligned[i] and
                     rsi_aligned[i-1] < rsi_aligned[i])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals