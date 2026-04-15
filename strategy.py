#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted RSI with 12h EMA Filter
# Uses RSI(14) on volume-weighted close to reduce noise, combined with 12h EMA50 trend filter.
# Long when VW-RSI < 30 and price above 12h EMA50, short when VW-RSI > 70 and price below 12h EMA50.
# Volume weighting filters out low-conviction moves, improving signal quality in both bull and bear markets.
# Discrete sizing (0.25) limits overtrading; target 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume-weighted close: (high + low + close) / 3, weighted by volume
    typical_price = (high + low + close) / 3.0
    vw_close = typical_price * volume
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # VW-RSI calculation
    delta = pd.Series(vw_close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    vw_rsi = rsi.values
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(vw_rsi[i]) or np.isnan(ema_12h_aligned[i])):
            continue
        
        # Long: VW-RSI < 30 (oversold) and price above 12h EMA50
        if vw_rsi[i] < 30 and close[i] > ema_12h_aligned[i]:
            signals[i] = 0.25
        
        # Short: VW-RSI > 70 (overbought) and price below 12h EMA50
        elif vw_rsi[i] > 70 and close[i] < ema_12h_aligned[i]:
            signals[i] = -0.25
        
        # Exit: VW-RSI returns to neutral range (40-60)
        elif i > 0 and (
            (signals[i-1] == 0.25 and vw_rsi[i] >= 40) or
            (signals[i-1] == -0.25 and vw_rsi[i] <= 60)
        ):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_VW_RSI_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0