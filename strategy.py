#!/usr/bin/env python3
name = "6h_PriceAction_1dMomentum_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for momentum and trend
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # 1h EMA for short-term trend
    close_1h = df_1h['close'].values
    ema10_1h = pd.Series(close_1h).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema30_1h = pd.Series(close_1h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema10_1h_aligned = align_htf_to_ltf(prices, df_1h, ema10_1h)
    ema30_1h_aligned = align_htf_to_ltf(prices, df_1h, ema30_1h)
    
    # 1h RSI for momentum
    delta = pd.Series(close_1h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1h = 100 - (100 / (1 + rs))
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h.values)
    
    # 6h price action: higher high/low for uptrend, lower high/low for downtrend
    hh = (high > np.roll(high, 1)) & (high > np.roll(high, 2))
    hl = (low > np.roll(low, 1)) & (low > np.roll(low, 2))
    lh = (high < np.roll(high, 1)) & (high < np.roll(high, 2))
    ll = (low < np.roll(low, 1)) & (low < np.roll(low, 2))
    
    # Volume filter: current volume > 1.5x 10-period average
    volume_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(ema10_1h_aligned[i]) or np.isnan(ema30_1h_aligned[i]) or 
            np.isnan(rsi_1h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Higher high/low (bullish price action) + EMA10 > EMA30 + RSI > 50 + volume
            if (hh[i] and hl[i] and 
                ema10_1h_aligned[i] > ema30_1h_aligned[i] and
                rsi_1h_aligned[i] > 50 and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lower high/low (bearish price action) + EMA10 < EMA30 + RSI < 50 + volume
            elif (lh[i] and ll[i] and 
                  ema10_1h_aligned[i] < ema30_1h_aligned[i] and
                  rsi_1h_aligned[i] < 50 and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown in price action (lower high/low) or EMA cross down
            if (lh[i] and ll[i]) or (ema10_1h_aligned[i] < ema30_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: recovery in price action (higher high/low) or EMA cross up
            if (hh[i] and hl[i]) or (ema10_1h_aligned[i] > ema30_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals