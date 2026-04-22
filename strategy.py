#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h momentum pullback with 4h trend filter (EMA200) and 1d volume confirmation
    # In strong trends, price pulls back to the 4h EMA200 before continuing.
    # This strategy buys dips in uptrends and sells rallies in downtrends.
    # Volume confirmation on 1d ensures institutional participation.
    # Session filter (08-20 UTC) reduces noise. Target: 15-30 trades/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 1h RSI(14) for pullback detection
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after RSI warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Pullback to 4h EMA200 in uptrend with RSI < 40 and above-average volume
            if (close[i] > ema200_4h_aligned[i] and  # Uptrend
                rsi[i] < 40 and                    # Pullback
                volume[i] > vol_ma20_1d_aligned[i]): # Volume confirmation
                signals[i] = 0.20
                position = 1
            # Short: Rally to 4h EMA200 in downtrend with RSI > 60 and above-average volume
            elif (close[i] < ema200_4h_aligned[i] and  # Downtrend
                  rsi[i] > 60 and                    # Rally
                  volume[i] > vol_ma20_1d_aligned[i]): # Volume confirmation
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1:
                if rsi[i] > 60 or close[i] < ema200_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi[i] < 40 or close[i] > ema200_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_EMA200_Pullback_RSI_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0