#!/usr/bin/env python3
"""
1h_HTF_Confluence_MeanReversion_v1
Hypothesis: On 1h timeframe, trade mean reversions at extreme RSI levels only when aligned with 4h and 1d trends. Uses 4h EMA50 for trend direction and 1d RSI(14) for regime filter. Entry when 1h RSI < 30 (oversold) in 4h uptrend AND 1d not overbought, or 1h RSI > 70 (overbought) in 4h downtrend AND 1d not oversold. Discrete position sizing (0.20) with ATR-based stoploss (2.0x) to limit drawdown. Target 15-37 trades/year by requiring HTF alignment and extreme RSI levels.
"""

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for regime filter (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d for regime filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 1h RSI(14) for entry signal
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_14_1h = 100 - (100 / (1 + rs_1h))
    
    # Calculate ATR(14) for stoploss on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(50), 1d RSI, 1h RSI, ATR
    start_idx = max(50, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(rsi_14_1h[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        rsi_1h_val = rsi_14_1h[i]
        rsi_1d_val = rsi_14_1d_aligned[i]
        trend_4h_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: 1h RSI < 30 (oversold) AND 4h uptrend AND 1d RSI not overbought (< 70)
            long_signal = (rsi_1h_val < 30) and trend_4h_up and (rsi_1d_val < 70)
            
            # Short: 1h RSI > 70 (overbought) AND 4h downtrend AND 1d RSI not oversold (> 30)
            short_signal = (rsi_1h_val > 70) and trend_4h_down and (rsi_1d_val > 30)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: 1h RSI > 50 (mean reversion complete) OR stoploss
            if (rsi_1h_val > 50) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: 1h RSI < 50 (mean reversion complete) OR stoploss
            if (rsi_1h_val < 50) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HTF_Confluence_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0