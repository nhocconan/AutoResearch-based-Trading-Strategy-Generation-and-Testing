#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h RSI(2) mean reversion with 1d EMA200 trend filter and volume confirmation
    # RSI(2) identifies extreme short-term reversals; EMA200 filters for trend direction
    # Long when RSI(2)<10 and price > EMA200; Short when RSI(2)>90 and price < EMA200
    # Volume spike confirms momentum behind the reversal
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after RSI warmup
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) + volume spike + price above 1d EMA200
            if rsi[i] < 10 and vol_spike[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) + volume spike + price below 1d EMA200
            elif rsi[i] > 90 and vol_spike[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral (50) or trend reversal vs 1d EMA200
            if position == 1:
                if rsi[i] > 50 or close[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] < 50 or close[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_RSI2_MeanReversion_1dEMA200_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0