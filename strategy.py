#!/usr/bin/env python3
# 6h_TRIX_RSI_Trend_1dVolFilter
# Hypothesis: 6h chart strategy combining TRIX (momentum) and RSI (mean reversion) with 1d trend filter and volume confirmation.
# TRIX crossing zero signals momentum shifts; RSI <30 or >70 identifies overextended conditions for reversal trades.
# 1d EMA50 filters trend direction to avoid counter-trend trades. Volume >1.5x average confirms momentum validity.
# Designed for low-frequency trading (12-30 trades/year) to minimize fee drag while capturing swings in both bull and bear markets.

timeframe = "6h"
name = "6h_TRIX_RSI_Trend_1dVolFilter"
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
    
    # Get 1d data for trend filter (EMA50) and volume reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX (15,9,9) on 6h close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) then percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_raw = pd.Series(ema3).pct_change() * 100  # percentage
    trix = trix_raw.fillna(0).values
    
    # Calculate RSI(14) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Volume spike detection: 1.5x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 15)  # Ensure we have TRIX/RSI/volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero (bullish momentum) AND RSI < 30 (oversold) AND volume confirmation AND 1d trend bullish
            if (trix[i] > 0 and trix[i-1] <= 0 and  # zero-cross up
                rsi[i] < 30 and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero (bearish momentum) AND RSI > 70 (overbought) AND volume confirmation AND 1d trend bearish
            elif (trix[i] < 0 and trix[i-1] >= 0 and  # zero-cross down
                  rsi[i] > 70 and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX turns negative (momentum loss) OR RSI > 70 (overbought)
            if trix[i] < 0 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX turns positive (momentum loss) OR RSI < 30 (oversold)
            if trix[i] > 0 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals