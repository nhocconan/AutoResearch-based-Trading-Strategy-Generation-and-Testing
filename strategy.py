#!/usr/bin/env python3
# 1h_triple_confirmation_4h1d_v1
# Hypothesis: Use 4h EMA(50) and 1d EMA(200) for trend direction, with 1h RSI(14) for entry timing.
# Long when 4h EMA(50) > 1d EMA(200) and RSI(14) < 30 (oversold bounce in uptrend).
# Short when 4h EMA(50) < 1d EMA(200) and RSI(14) > 70 (overbought rejection in downtrend).
# Exit when RSI returns to neutral (40-60 range) or opposite signal.
# Designed to work in both bull and bear markets by using higher timeframe trend filters.
# Target: 20-40 trades/year to minimize fee drag while capturing high-probability mean-reversion entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_triple_confirmation_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # === 4h Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # === 1h Entry Signal: RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_4h_50_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or \
           np.isnan(rsi[i]) or not session_filter[i]:
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction: 4h EMA50 vs 1d EMA200
        uptrend = ema_4h_50_aligned[i] > ema_1d_200_aligned[i]
        downtrend = ema_4h_50_aligned[i] < ema_1d_200_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or opposite signal
            if rsi[i] >= 40 or rsi[i] <= 60:  # Actually, exit when RSI >= 40 (recovery from oversold)
                if rsi[i] >= 40:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or opposite signal
            if rsi[i] <= 60 or rsi[i] >= 40:  # Exit when RSI <= 60 (recovery from overbought)
                if rsi[i] <= 60:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: uptrend + RSI oversold (<30)
            if uptrend and rsi[i] < 30:
                position = 1
                signals[i] = 0.20
            # Short entry: downtrend + RSI overbought (>70)
            elif downtrend and rsi[i] > 70:
                position = -1
                signals[i] = -0.20
    
    return signals