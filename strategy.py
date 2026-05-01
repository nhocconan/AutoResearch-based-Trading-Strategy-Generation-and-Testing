#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter.
# Long when RSI < 30 AND price > 4h EMA50 (uptrend) AND hour between 08-20 UTC.
# Short when RSI > 70 AND price < 4h EMA50 (downtrend) AND hour between 08-20 UTC.
# Uses discrete sizing 0.20 to manage drawdown and minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# 4h EMA50 provides higher timeframe trend direction to avoid counter-trend trades.
# Session filter reduces noise during low-volume off-hours.
# RSI extremes provide mean reversion entries in ranging markets.

name = "1h_RSI14_MeanRev_4hEMA50Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate RSI(14) on 1h close prices
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        curr_ema = ema_50_aligned[i]
        curr_in_session = in_session[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: RSI oversold AND price above 4h EMA50 AND in session
            if (curr_rsi < 30 and 
                curr_close > curr_ema and 
                curr_in_session):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought AND price below 4h EMA50 AND in session
            elif (curr_rsi > 70 and 
                  curr_close < curr_ema and 
                  curr_in_session):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI returns to neutral (50) OR price crosses below 4h EMA50
            if (curr_rsi >= 50 or 
                curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) OR price crosses above 4h EMA50
            if (curr_rsi <= 50 or 
                curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals