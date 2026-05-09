#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 34-period EMA trend filter with 1d RSI(14) mean reversion and volume confirmation.
# Uses daily RSI for contrarian entries when extreme, 4h EMA for trend alignment,
# and volume surge for confirmation. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 20-50 trades/year to avoid fee drag.
name = "4h_EMA34_1dRSI14_MeanRev_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI mean reversion filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI for daily timeframe
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (1 - alpha) * avg_gain[i-1] + alpha * gain[i]
        avg_loss[i] = (1 - alpha) * avg_loss[i-1] + alpha * loss[i]
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 34-period EMA for 4h timeframe
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume > 1.8x 34-period EMA (moderate threshold)
    vol_ema34 = pd.Series(volume).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_confirm = volume > (1.8 * vol_ema34)
    
    # Align 1d RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need 34 periods for EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema34[i]) or np.isnan(vol_ema34[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price > EMA34 (uptrend) + RSI < 30 (oversold) + volume surge
            if (price > ema34[i] and rsi_aligned[i] < 30 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < EMA34 (downtrend) + RSI > 70 (overbought) + volume surge
            elif (price < ema34[i] and rsi_aligned[i] > 70 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < EMA34 or RSI > 50 (mean reversion)
            if price < ema34[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > EMA34 or RSI < 50 (mean reversion)
            if price > ema34[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals