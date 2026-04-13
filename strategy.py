#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w EMA trend filter and RSI mean reversion.
# Long: Price below weekly EMA50 AND RSI(14) < 30 (oversold in bearish trend)
# Short: Price above weekly EMA50 AND RSI(14) > 70 (overbought in bullish trend)
# Uses weekly EMA for trend direction and daily RSI for mean-reversion entries.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50
    ema_50_1w = np.full(len(close_1w), np.nan)
    close_1w_series = pd.Series(close_1w)
    ema_50_1w_series = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w = ema_50_1w_series.values
    
    # Align weekly EMA50 to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # Start with neutral RSI
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_1w_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: Price below weekly EMA50 AND RSI < 30 (oversold in downtrend)
            if price < ema_50 and rsi_val < 30:
                position = 1
                signals[i] = position_size
            # Short: Price above weekly EMA50 AND RSI > 70 (overbought in uptrend)
            elif price > ema_50 and rsi_val > 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses above weekly EMA50 OR RSI > 70
            if price > ema_50 or rsi_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses below weekly EMA50 OR RSI < 30
            if price < ema_50 or rsi_val < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA50_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0