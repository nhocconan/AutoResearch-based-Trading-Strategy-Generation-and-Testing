#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-period RSI with 4h/1d trend filter and volume confirmation.
# Long when RSI(4) < 30 (oversold) in uptrend (price > 4h EMA50 > 1d EMA200) with volume spike.
# Short when RSI(4) > 70 (overbought) in downtrend (price < 4h EMA50 < 1d EMA200) with volume spike.
# Designed to capture mean reversions in both bull and bear markets by following higher timeframe trends.
# RSI(4) is sensitive for timely entries; 4h/1d EMA filters ensure trend alignment.
# Volume confirmation avoids low-conviction moves.
name = "1h_RSI4_4h1dEMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter and entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # RSI(4) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 trend filter
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA200 trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI(4) < 30 (oversold) + price > 4h EMA50 > 1d EMA200 + volume
            if (rsi[i] < 30 and price > ema_4h_aligned[i] and ema_4h_aligned[i] > ema_1d_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI(4) > 70 (overbought) + price < 4h EMA50 < 1d EMA200 + volume
            elif (rsi[i] > 70 and price < ema_4h_aligned[i] and ema_4h_aligned[i] < ema_1d_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI(4) > 70 (overbought) or price < 4h EMA50
            if rsi[i] > 70 or price < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI(4) < 30 (oversold) or price > 4h EMA50
            if rsi[i] < 30 or price > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals