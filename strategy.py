#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h trend direction + 1h session-specific mean reversion entries.
# Uses 4h EMA(50) for trend filter (bullish when close > EMA, bearish when close < EMA).
# Enters on 1h during 08-20 UTC: long on RSI(14) < 30 pullback in bullish 4h trend,
# short on RSI(14) > 70 rally in bearish 4h trend.
# Exits on RSI crossing 50 (mean reversion completion) or 4h trend reversal.
# Discrete size 0.20. Target: 60-150 total trades over 4 years (15-37/year).
# Designed to work in both bull (buy dips) and bear (sell rallies) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h Indicators: EMA(50) for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    bullish_4h = close_4h > ema_4h  # 4h bullish when close > EMA50
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h.astype(float))
    
    # === 1h Indicators: RSI(14) for mean reversion entries ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure RSI and EMA are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if outside session or missing data
        if not session_filter[i] or np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(bullish_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        rsi_val = rsi[i]
        is_bullish_4h = bullish_4h_aligned[i] > 0.5
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI crosses above 50 (mean reversion complete) OR 4h trend turns bearish
            if rsi_val > 50 or not is_bullish_4h:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI crosses below 50 (mean reversion complete) OR 4h trend turns bullish
            if rsi_val < 50 or is_bullish_4h:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30 (oversold) in bullish 4h trend
            if rsi_val < 30 and is_bullish_4h:
                signals[i] = 0.20
                position = 1
            
            # SHORT: RSI > 70 (overbought) in bearish 4h trend
            elif rsi_val > 70 and not is_bullish_4h:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "4h_EMA50_1hRSI_MeanReversion_Session_V1"
timeframe = "1h"
leverage = 1.0