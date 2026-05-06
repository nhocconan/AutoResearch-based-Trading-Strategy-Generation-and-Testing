#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session timing
# Long when 1h RSI < 30 AND 4h close > 4h EMA50 AND hour between 08-20 UTC
# Short when 1h RSI > 70 AND 4h close < 4h EMA50 AND hour between 08-20 UTC
# Exit when 1h RSI crosses 50 (mean reversion completion)
# Uses discrete sizing 0.20 to minimize fee drag
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# RSI mean reversion works in ranging markets; 4h EMA50 filter ensures alignment with higher timeframe trend
# Session filter (08-20 UTC) reduces noise during low-liquidity periods, improving win rate
# Works in both bull and bear markets by combining mean reversion entries with trend filter

name = "1h_RSI14_MeanRev_4hEMA50_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1h RSI(14) and 4h EMA50 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(close) < 14 or len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed bars)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI oversold AND 4h uptrend
            if rsi_values[i] < 30 and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought AND 4h downtrend
            elif rsi_values[i] > 70 and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion complete)
            if rsi_values[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion complete)
            if rsi_values[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals