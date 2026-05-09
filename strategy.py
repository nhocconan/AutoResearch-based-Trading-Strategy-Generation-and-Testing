#!/usr/bin/env python3
# Hypothesis: 1h RSI(14) mean reversion with 4h EMA(50) trend filter and volume confirmation
# Long when: RSI < 30, price > 4h EMA(50) (bullish bias), volume > 1.5x 20-period average
# Short when: RSI > 70, price < 4h EMA(50) (bearish bias), volume > 1.5x 20-period average
# Exit when: RSI crosses above 50 (long exit) or below 50 (short exit)
# Position size: 0.20 (20% of capital) to limit drawdown. Target: 15-37 trades/year (60-150 total over 4 years).
# Uses 4h EMA for trend direction (reduces whipsaw) and 1h RSI for precise entry/exit timing.
# Session filter (08-20 UTC) avoids low-liquidity periods. Works in both bull (mean reversion in uptrend) and bear (mean reversion in downtrend).

name = "1h_RSI14_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain_ma / loss_ma
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close']
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_spike[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 (oversold), price > 4h EMA(50) (bullish bias), volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought), price < 4h EMA(50) (bearish bias), volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 (momentum shift)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses below 50 (momentum shift)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals