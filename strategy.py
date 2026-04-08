#!/usr/bin/env python3
# 1d_market_regime_reversion_v1
# Hypothesis: On daily timeframe, combine mean reversion in ranging markets with trend following in trending markets using Choppiness Index as regime filter.
# In ranging markets (CHOP > 61.8): Buy near support (low + 0.3 * ATR), sell near resistance (high - 0.3 * ATR).
# In trending markets (CHOP < 38.2): Follow trend using EMA(50) direction.
# Avoids whipsaw in strong trends and captures reversals in ranges. Low trade frequency (~10-20/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_market_regime_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index (14)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            if chop[i] > 61.8:  # Ranging market - mean reversion exit
                # Exit near resistance
                resistance = high[i] - 0.3 * atr[i]
                if close[i] >= resistance:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Trending market - trend following exit
                # Exit when trend changes
                if close[i] < ema_50_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop[i] > 61.8:  # Ranging market - mean reversion exit
                # Exit near support
                support = low[i] + 0.3 * atr[i]
                if close[i] <= support:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Trending market - trend following exit
                # Exit when trend changes
                if close[i] > ema_50_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            if chop[i] > 61.8:  # Ranging market - mean reversion entry
                # Buy near support, sell near resistance
                support = low[i] + 0.3 * atr[i]
                resistance = high[i] - 0.3 * atr[i]
                
                if close[i] <= support:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= resistance:
                    position = -1
                    signals[i] = -0.25
            else:  # Trending market - trend following entry
                # Follow weekly EMA50 trend
                if close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals