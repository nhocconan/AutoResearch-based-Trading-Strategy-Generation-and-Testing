#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h/1d trend filters to reduce noise and trade frequency.
# Uses 4h RSI(14) for medium-term trend and 1d EMA(200) for long-term bias.
# Entry: RSI(14) < 30 (oversold) + close > 1d EMA200 (uptrend bias) for long.
# Entry: RSI(14) > 70 (overbought) + close < 1d EMA200 (downtrend bias) for short.
# Exit: Opposite RSI cross (50 level) or trend bias violation.
# Uses session filter (08-20 UTC) to avoid low-volume Asian session noise.
# Fixed position size 0.20 to limit risk and reduce churn.
# Target: 15-30 trades/year per symbol by requiring multiple confluence factors.

name = "1h_RSI_EMA200_TrendFilter_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h RSI(14) for trend filter
    df_4h = get_htf_data(prices, '4h')
    delta_4h = pd.Series(df_4h['close']).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d EMA(200) for long-term trend
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 has enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold + 4h RSI > 50 (bullish bias) + close > 1d EMA200
            if (rsi[i] < 30 and 
                rsi_4h_aligned[i] > 50 and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 4h RSI < 50 (bearish bias) + close < 1d EMA200
            elif (rsi[i] > 70 and 
                  rsi_4h_aligned[i] < 50 and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if RSI > 50 or trend bias fails
            if (rsi[i] > 50) or (close[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if RSI < 50 or trend bias fails
            if (rsi[i] < 50) or (close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals