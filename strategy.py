#!/usr/bin/env python3
# Hypothesis: 12h RSI(14) mean reversion with 1w EMA50 trend filter and volume confirmation
# Long when RSI<30 and price > 1w EMA50 and volume > 1.5x average
# Short when RSI>70 and price < 1w EMA50 and volume > 1.5x average
# Exit when RSI returns to neutral zone (40-60) or opposite extreme
# Uses weekly trend to avoid counter-trend trades, RSI for mean reversion, volume for conviction
# Designed for low-frequency, high-conviction trades in ranging markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_RSI14_MeanReversion_1wEMA50_Volume"
timeframe = "12h"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for RSI and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold, price above weekly EMA, volume spike
            if (rsi[i] < 30 and 
                close[i] > ema50_1w_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought, price below weekly EMA, volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or becomes overbought
            if (rsi[i] >= 40 and rsi[i] <= 60) or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or becomes oversold
            if (rsi[i] >= 40 and rsi[i] <= 60) or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals