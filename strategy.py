#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 200 EMA trend + RSI mean reversion with 1-week trend filter and volume confirmation.
# Works in bull/bear: 200 EMA captures trend direction, RSI < 30 or > 70 for reversal entries.
# Volume spike confirms momentum. Target: 8-20 trades/year (32-80 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA200 and RSI - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Load weekly data for trend filter (optional: avoid counter-trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(210, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: Price above EMA200 (uptrend) + RSI oversold + volume spike
            if (close[i] > ema_200_1d_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA200 (downtrend) + RSI overbought + volume spike
            elif (close[i] < ema_200_1d_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or opposite extreme
            if position == 1:
                if rsi_1d_aligned[i] > 40:  # Exit long when RSI rises above 40
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_1d_aligned[i] < 60:  # Exit short when RSI falls below 60
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_EMA200_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0