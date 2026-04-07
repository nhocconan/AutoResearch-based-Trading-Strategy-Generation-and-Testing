# 1h RSI Pullback with Multi-Timeframe Trend Filter
# Strategy: Buy pullbacks in uptrends, sell rallies in downtrends
# Uses 1d trend (EMA200) and 4h momentum (RSI) for direction, 1h for entry timing
# RSI < 40 in uptrend for long, RSI > 60 in downtrend for short
# Target: 15-35 trades/year, low frequency to avoid fee drag
# Works in bull/bear by following higher timeframe trend

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Multi-timeframe trend filters (calculated once) ===
    # 1d trend: EMA200
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h momentum: RSI(14)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # === 1h RSI for entry timing ===
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.maximum(delta_1h, 0)
    loss_1h = np.maximum(-delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_14_1h = 100 - (100 / (1 + rs_1h))
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or 
            np.isnan(rsi_14_1h[i])):
            continue
        
        # Determine trend from higher timeframes
        # Uptrend: price above 1d EMA200 AND 4h RSI > 50
        # Downtrend: price below 1d EMA200 AND 4h RSI < 50
        is_uptrend = close[i] > ema_200_1d_aligned[i] and rsi_14_4h_aligned[i] > 50
        is_downtrend = close[i] < ema_200_1d_aligned[i] and rsi_14_4h_aligned[i] < 50
        
        # Entry logic: pullbacks in trend
        if is_uptrend and rsi_14_1h[i] < 40:
            # Long on RSI pullback in uptrend
            signals[i] = 0.20
        elif is_downtrend and rsi_14_1h[i] > 60:
            # Short on RSI bounce in downtrend
            signals[i] = -0.20
        else:
            # No clear signal, stay flat
            signals[i] = 0.0
    
    return signals