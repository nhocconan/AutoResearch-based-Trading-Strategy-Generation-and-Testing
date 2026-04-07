#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h Trend Filter and Session Filter
# Hypothesis: In strong trends (4h EMA21 > EMA50), pullbacks to RSI(14) < 30 offer high-probability long entries.
# In weak trends (4h EMA21 < EMA50), pullbacks to RSI(14) > 70 offer high-probability short entries.
# Restricts trading to 08:00-20:00 UTC to avoid low-liquidity periods. Uses 1h timeframe for precise entry timing.
# Target: 15-37 trades/year (60-150 over 4 years) by requiring strong 4h trend alignment + extreme RSI + session filter.
name = "1h_rsi_pullback_4h_trend_session_v1"
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
    volume = prices['volume'].values
    
    # Get 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # RSI(14) on 1h timeframe
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA21 and EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # Pre-compute hour from DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        hour = hours[i]
        if hour < 8 or hour > 20:  # Outside 08:00-20:00 UTC
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI reaches 50 (mean reversion) or 4h trend turns bearish
            if rsi[i] >= 50 or ema21_4h_aligned[i] < ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: RSI reaches 50 (mean reversion) or 4h trend turns bullish
            if rsi[i] <= 50 or ema21_4h_aligned[i] > ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Strong bullish trend: EMA21 > EMA50 on 4h
            if ema21_4h_aligned[i] > ema50_4h_aligned[i]:
                # Look for pullback: RSI < 30 (oversold)
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.20  # Enter long
            # Strong bearish trend: EMA21 < EMA50 on 4h
            elif ema21_4h_aligned[i] < ema50_4h_aligned[i]:
                # Look for pullback: RSI > 70 (overbought)
                if rsi[i] > 70:
                    position = -1
                    signals[i] = -0.20  # Enter short
    
    return signals