#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX Trend + Volume Spike + 12h EMA Filter
# Use ADX(14) > 25 to identify trending markets, volume > 2x 20-bar median for confirmation,
# and 12h EMA50 for higher-timeframe trend alignment. Long when +DI > -DI and price above 12h EMA50,
# short when -DI > +DI and price below 12h EMA50. Uses discrete sizing (0.25) to limit overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # ADX calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: ADX > 25, +DI > -DI, volume spike, price above 12h EMA50
        if (adx[i] > 25 and plus_di[i] > minus_di[i] and 
            volume[i] > vol_threshold[i] and close[i] > ema_12h_aligned[i]):
            signals[i] = 0.25
        
        # Short: ADX > 25, -DI > +DI, volume spike, price below 12h EMA50
        elif (adx[i] > 25 and minus_di[i] > plus_di[i] and 
              volume[i] > vol_threshold[i] and close[i] < ema_12h_aligned[i]):
            signals[i] = -0.25
        
        # Exit: ADX weakens or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (adx[i] <= 25 or plus_di[i] <= minus_di[i] or close[i] <= ema_12h_aligned[i])) or
               (signals[i-1] == -0.25 and (adx[i] <= 25 or minus_di[i] <= plus_di[i] or close[i] >= ema_12h_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_ADX_Trend_Volume_12hEMA"
timeframe = "4h"
leverage = 1.0