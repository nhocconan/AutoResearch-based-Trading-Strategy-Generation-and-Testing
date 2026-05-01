#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter (08-20 UTC).
# Uses 4h EMA50 for trend direction (long when price > EMA50, short when price < EMA50).
# Enters on RSI extremes (<30 for long, >70 for short) only during active session (08-20 UTC).
# Exits on RSI return to neutral (40-60 range) or trend reversal.
# Designed for low trade frequency (target: 15-37/year) by combining tight RSI extremes,
# session filter, and 4h trend alignment to avoid chop and false signals.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "1h_RSI_MeanReversion_4hEMA50_Trend_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for RSI and EMA50
    start_idx = max(14, 50)  # 50
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        
        # Trend filter: 4h EMA50 direction
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # RSI conditions
        rsi_oversold = curr_rsi < 30
        rsi_overbought = curr_rsi > 70
        rsi_neutral = (curr_rsi >= 40) & (curr_rsi <= 60)
        
        if position == 0:  # Flat - look for new entries
            # Long: RSI oversold AND uptrend
            if rsi_oversold and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND downtrend
            elif rsi_overbought and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on RSI return to neutral OR trend reversal to downtrend
            if rsi_neutral or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on RSI return to neutral OR trend reversal to uptrend
            if rsi_neutral or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals