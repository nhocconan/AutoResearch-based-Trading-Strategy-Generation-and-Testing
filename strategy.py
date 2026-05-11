#!/usr/bin/env python3
"""
4h_RSI_Extreme_Confluence_v1
Hypothesis: Extreme RSI readings on 4h (below 25 or above 75) combined with 12h EMA trend and volume spike provide high-probability mean-reversion entries in both bull and bear markets.
- RSI < 25: oversold, potential long
- RSI > 75: overbought, potential short
- 12h EMA50 determines trend direction for bias (long only in uptrend, short only in downtrend)
- Volume > 1.5x 20-period average confirms participation
- Designed for low trade frequency (~20-30 trades/year) by requiring multiple confluence factors
- Works in bull markets via pullbacks to EMA in uptrend, works in bear markets via bounces in downtrend
"""

name = "4h_RSI_Extreme_Confluence_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- RSI (14-period) on 4h close ---
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # --- EMA50 on 12h close ---
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume vs 20-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ratio[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend bias from 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Look for extreme RSI with trend bias and volume confirmation
            if rsi[i] < 25 and uptrend and vol_spike:  # Oversold in uptrend -> long
                signals[i] = 0.25
                position = 1
            elif rsi[i] > 75 and downtrend and vol_spike:  # Overbought in downtrend -> short
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral or opposite extreme
            if position == 1:
                # Exit long: RSI > 50 or RSI > 70 (overbought)
                exit_signal = (rsi[i] > 50) or (rsi[i] > 70)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 50 or RSI < 30 (oversold)
                exit_signal = (rsi[i] < 50) or (rsi[i] < 30)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals