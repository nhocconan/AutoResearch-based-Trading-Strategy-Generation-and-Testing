#!/usr/bin/env python3
# Hypothesis: 1h RSI mean reversion with 4h trend filter and session timing.
# Long when: RSI(14) < 30 (oversold) AND 4h close > 4h EMA50 (uptrend) AND hour in 08-20 UTC.
# Short when: RSI(14) > 70 (overbought) AND 4h close < 4h EMA50 (downtrend) AND hour in 08-20 UTC.
# Exit when: RSI returns to neutral zone (40-60) or opposite extreme reached.
# Uses discrete position sizing (0.20) to limit fee churn. Designed for 1h timeframe.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Works in bull markets by buying oversold dips in uptrends, in bear markets by selling overbought rallies in downtrends.

name = "1h_RSI_MeanReversion_4hEMA50_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
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
    
    # Pre-compute session hours
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        in_session = 8 <= hours[i] <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (oversold) AND 4h close > 4h EMA50 (uptrend)
            if (rsi_values[i] < 30 and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 (overbought) AND 4h close < 4h EMA50 (downtrend)
            elif (rsi_values[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>=40) or becomes overbought (>70)
            if rsi_values[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<=60) or becomes oversold (<30)
            if rsi_values[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals