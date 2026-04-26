#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrendFilter
Hypothesis: On 1h timeframe, use RSI(14) for mean reversion entries (RSI<30 long, RSI>70 short) 
but only when aligned with 4h trend (price > 4h EMA50 for longs, price < 4h EMA50 for shorts).
Add session filter (08-20 UTC) to avoid low-volume hours. Uses discrete sizing (0.20) to minimize fee drag.
Target: 60-120 trades over 4 years (15-30/year) on 1h. Uses 4h EMA for trend filter to avoid SOL-only bias.
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
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 14 for RSI, 50 for EMA)
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            # Hold current position outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        rsi_val = rsi[i]
        close_val = close[i]
        ema_val = ema_50_4h_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(rsi_val) or np.isnan(ema_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long condition: RSI < 30 (oversold) + price above 4h EMA50 (uptrend)
        long_condition = (rsi_val < 30) and (close_val > ema_val)
        # Short condition: RSI > 70 (overbought) + price below 4h EMA50 (downtrend)
        short_condition = (rsi_val > 70) and (close_val < ema_val)
        
        # Exit conditions: RSI reverts to mean (40-60) or opposite extreme
        exit_long = rsi_val > 50
        exit_short = rsi_val < 50
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_RSI_MeanReversion_4hTrendFilter"
timeframe = "1h"
leverage = 1.0