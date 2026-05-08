#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Relative Strength Index (RSI) with 12-hour trend filter and volume confirmation
# Long when RSI < 30 (oversold), 12h EMA(20) uptrend, and volume spike
# Short when RSI > 70 (overbought), 12h EMA(20) downtrend, and volume spike
# RSI identifies mean-reversion opportunities; 12h EMA provides higher timeframe bias
# Volume spike confirms institutional participation; avoids choppy false reversals
# Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

name = "4h_RSI_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12-hour data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_12h_val = ema20_12h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold), 12h uptrend, and volume spike
            if rsi_val < 30 and close[i] > ema20_12h_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought), 12h downtrend, and volume spike
            elif rsi_val > 70 and close[i] < ema20_12h_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (neutral) or 12h trend turns down
            if rsi_val > 50 or close[i] < ema20_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (neutral) or 12h trend turns up
            if rsi_val < 50 or close[i] > ema20_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals