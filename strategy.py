#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_ema_retracement_v1
# In trending markets, price often retraces to EMA(21) on 1d chart before continuing.
# This strategy enters long when price pulls back to EMA(21) on 1d during uptrend (EMA(50) > EMA(200)),
# and short when price rallies to EMA(21) during downtrend (EMA(50) < EMA(200)).
# Uses volume confirmation and volatility filter (ATR) to avoid false signals.
# Designed for low trade frequency (~25-40/year) with high win rate in trends.
name = "4h_1d_ema_retracement_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMAs on 1d
    close_1d = df_1d['close'].values
    ema21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 4h timeframe
    ema21_4h = align_htf_to_ltf(prices, df_1d, ema21)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50)
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Volatility filter: avoid extremely low volatility (ATR < 0.5 * 50-period avg ATR)
    # True range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > (atr_ma * 0.5)  # sufficient volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if EMAs not ready
        if np.isnan(ema21_4h[i]) or np.isnan(ema50_4h[i]) or np.isnan(ema200_4h[i]):
            signals[i] = 0.0
            continue
        
        # Check filters
        if not (vol_confirm[i] and vol_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on EMA50 vs EMA200
        uptrend = ema50_4h[i] > ema200_4h[i]
        downtrend = ema50_4h[i] < ema200_4h[i]
        
        # Long setup: uptrend + price near EMA21 (within 0.5*ATR)
        if uptrend and position != 1:
            if abs(close[i] - ema21_4h[i]) <= (0.5 * atr[i]):
                position = 1
                signals[i] = 0.25
            else:
                # Hold flat or reverse
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        # Short setup: downtrend + price near EMA21 (within 0.5*ATR)
        elif downtrend and position != -1:
            if abs(close[i] - ema21_4h[i]) <= (0.5 * atr[i]):
                position = -1
                signals[i] = -0.25
            else:
                # Hold flat or reverse
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        # Exit: trend reversal
        elif (uptrend and position == -1) or (downtrend and position == 1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals