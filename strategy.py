#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(14) with 1w EMA34 trend filter and volume confirmation
# Long when RSI < 30 (oversold), 1w EMA34 rising, volume > 1.3x average
# Short when RSI > 70 (overbought), 1w EMA34 falling, volume > 1.3x average
# Uses RSI for mean reversion signals, weekly EMA for trend filter, volume for confirmation
# Targets 7-25 trades per year (28-100 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter avoiding counter-trend trades

name = "1d_RSI14_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate RSI(14) - mean reversion signal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA34 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need at least 34 days of data for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: RSI oversold (<30), 1w uptrend, volume confirmation
            if rsi_val < 30 and ema34_1w_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70), 1w downtrend, volume confirmation
            elif rsi_val > 70 and ema34_1w_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought (>70) or 1w trend down
            if rsi_val > 70 or ema34_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold (<30) or 1w trend up
            if rsi_val < 30 or ema34_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals