#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend(10,3) for trend direction and 1h RSI(14) with Bollinger Bands(20,2) for mean reversion entries
# Long when 4h uptrend + price touches lower BB and RSI<30 (oversold in uptrend)
# Short when 4h downtrend + price touches upper BB and RSI>70 (overbought in downtrend)
# Uses 4h trend filter to avoid counter-trend trades and reduce whipsaws
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete position sizing: 0.20 for entries, 0.0 for exit
# Target: 15-30 trades/year (~60-120 over 4 years) to minimize fee drag
# Works in bull markets via buying dips in uptrends and in bear markets via selling rallies in downtrends

name = "1h_4hSupertrend_1hRSIBB_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Supertrend(ATR=10, mult=3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2_4h = (high_4h + low_4h) / 2
    upper_basic_4h = hl2_4h + 3 * atr_4h
    lower_basic_4h = hl2_4h - 3 * atr_4h
    
    # Final Upper and Lower Bands
    final_upper_4h = np.zeros_like(close_4h)
    final_lower_4h = np.zeros_like(close_4h)
    final_upper_4h[0] = upper_basic_4h[0]
    final_lower_4h[0] = lower_basic_4h[0]
    
    for i in range(1, len(close_4h)):
        if close_4h[i-1] <= final_upper_4h[i-1]:
            final_upper_4h[i] = min(upper_basic_4h[i], final_upper_4h[i-1])
        else:
            final_upper_4h[i] = upper_basic_4h[i]
            
        if close_4h[i-1] >= final_lower_4h[i-1]:
            final_lower_4h[i] = max(lower_basic_4h[i], final_lower_4h[i-1])
        else:
            final_lower_4h[i] = lower_basic_4h[i]
    
    # Supertrend
    supertrend_4h = np.zeros_like(close_4h)
    uptrend_4h = np.zeros_like(close_4h, dtype=bool)
    supertrend_4h[0] = final_lower_4h[0]
    uptrend_4h[0] = True
    
    for i in range(1, len(close_4h)):
        if close_4h[i] <= final_upper_4h[i]:
            supertrend_4h[i] = final_upper_4h[i]
            uptrend_4h[i] = True
        elif close_4h[i] >= final_lower_4h[i]:
            supertrend_4h[i] = final_lower_4h[i]
            uptrend_4h[i] = False
        else:
            if uptrend_4h[i-1]:
                supertrend_4h[i] = final_upper_4h[i]
                uptrend_4h[i] = True
            else:
                supertrend_4h[i] = final_lower_4h[i]
                uptrend_4h[i] = False
    
    # Align 4h Supertrend and uptrend to 1h timeframe
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))  # align as float, then convert to bool
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h Bollinger Bands(20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 4h Supertrend and 1h BB
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # exit if outside session
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_rsi = rsi[i]
        curr_lower_bb = lower_bb[i]
        curr_upper_bb = upper_bb[i]
        curr_supertrend = supertrend_4h_aligned[i]
        curr_uptrend = uptrend_4h_aligned[i] > 0.5  # convert back to bool
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h uptrend + price touches lower BB + RSI oversold
            if curr_uptrend and curr_low <= curr_lower_bb and curr_rsi < 30:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + price touches upper BB + RSI overbought
            elif not curr_uptrend and curr_high >= curr_upper_bb and curr_rsi > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit: price crosses above Supertrend (trend change) or RSI > 70 (overbought)
            if curr_close >= curr_supertrend or curr_rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses below Supertrend (trend change) or RSI < 30 (oversold)
            if curr_close <= curr_supertrend or curr_rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals