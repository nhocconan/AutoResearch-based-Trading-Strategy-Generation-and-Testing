#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Long when RSI < 30 (oversold) and 4h trend up (EMA50 > EMA200) and volume spike
# Short when RSI > 70 (overbought) and 4h trend down (EMA50 < EMA200) and volume spike
# Uses 4h for directional bias (trend filter) and 1h for entry timing (RSI extremes)
# Volume spike confirms institutional participation; avoids false signals in chop
# Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost
# Includes session filter (08-20 UTC) to reduce noise trades outside active hours

name = "1h_RSI_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 and EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for RSI and EMAs
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50 = ema50_4h_aligned[i]
        ema200 = ema200_4h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold), 4h uptrend (EMA50 > EMA200), volume spike, in session
            if rsi_val < 30 and ema50 > ema200 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought), 4h downtrend (EMA50 < EMA200), volume spike, in session
            elif rsi_val > 70 and ema50 < ema200 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or 4h trend turns down
            if rsi_val > 70 or ema50 < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or 4h trend turns up
            if rsi_val < 30 or ema50 > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals