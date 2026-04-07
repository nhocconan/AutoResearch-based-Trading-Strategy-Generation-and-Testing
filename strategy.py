#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour RSI(2) mean reversion with daily trend filter and volume confirmation
# Long when RSI(2) < 10 on 6h, price > 200 EMA on 1d, and volume > 1.5x average
# Short when RSI(2) > 90 on 6h, price < 200 EMA on 1d, and volume > 1.5x average
# Exit when RSI(2) crosses above 50 (long) or below 50 (short)
# Uses extreme short-term RSI for mean reversion in trending markets, filtered by daily trend
# Position size: 0.25, designed for high-probability mean reversion trades

name = "6s_rsi2_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on daily close
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h RSI(2)
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # 6h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(rsi_2[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI(2) crosses above 50
            if rsi_2[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI(2) crosses below 50
            if rsi_2[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: extreme RSI(2) with volume confirmation and trend filter
            # Long: RSI(2) < 10, price > 200 EMA daily, volume spike
            if (rsi_2[i] < 10 and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90, price < 200 EMA daily, volume spike
            elif (rsi_2[i] > 90 and
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals