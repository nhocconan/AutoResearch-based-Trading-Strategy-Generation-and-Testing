#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) with 4-hour trend filter and volume confirmation
# Long when RSI < 30 (oversold) and 4-hour EMA(50) uptrend and volume spike
# Short when RSI > 70 (overbought) and 4-hour EMA(50) downtrend and volume spike
# Uses 4-hour EMA for trend direction (aligned properly), 1-hour RSI for entry timing
# Volume spike confirms momentum; avoids counter-trend entries in chop
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost
# Works in bull via mean reversion in uptrend, works in bear via mean reversion in downtrend

name = "1h_RSI_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4-hour data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1-hour RSI(14)
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) and 4h uptrend and volume spike
            if rsi_val < 30 and close[i] > ema50_4h_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought) and 4h downtrend and volume spike
            elif rsi_val > 70 and close[i] < ema50_4h_val and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or 4h trend turns down
            if rsi_val > 50 or close[i] < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or 4h trend turns up
            if rsi_val < 50 or close[i] > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals