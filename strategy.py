#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly trend filter, daily momentum, and volume confirmation.
# Long when: price > weekly VWAP AND daily RSI(14) > 55 AND 6h volume > 20-period average
# Short when: price < weekly VWAP AND daily RSI(14) < 45 AND 6h volume > 20-period average
# Exit when daily RSI crosses 50.
# Weekly VWAP acts as dynamic support/resistance, daily RSI provides momentum,
# volume confirms institutional interest. Designed for 6h timeframe with target 20-40 trades/year.
# Weekly trend filter adapts to bull/bear markets, volume filter reduces whipsaws.

name = "6h_WeeklyVWAP_DailyRSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Weekly VWAP for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate VWAP for each weekly bar
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w = vwap_1w.values
    
    # Align weekly VWAP to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_daily[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > weekly VWAP AND daily RSI > 55 AND volume confirmation
            long_condition = (close[i] > vwap_1w_aligned[i]) and (rsi_daily[i] > 55) and volume_filter[i]
            # Short: price < weekly VWAP AND daily RSI < 45 AND volume confirmation
            short_condition = (close[i] < vwap_1w_aligned[i]) and (rsi_daily[i] < 45) and volume_filter[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: daily RSI < 50
            if rsi_daily[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: daily RSI > 50
            if rsi_daily[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals