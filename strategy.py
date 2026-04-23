#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted Average Price (VWAP) Deviation with 1w Trend Filter and ATR Regime
- Price deviation from VWAP (6h) indicates short-term overextension; mean reversion expected
- 1w EMA200 defines primary trend: long only when price > EMA200, short only when price < EMA200
- ATR(14) regime filter: trade only when ATR > 0.5 * ATR(50) to avoid low-volatility chop
- Works in bull via long mean-reversion from VWAP support during uptrends
- Works in bear via short mean-reversion from VWAP resistance during downtrends
- Designed for 6h timeframe with low trade frequency (target: 12-37 trades/year) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate ATR(14) and ATR(50) for volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar: no previous close
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 200)  # need ATR14, ATR50, 1w EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: only trade when ATR14 > 0.5 * ATR50 (avoid low-vol chop)
        volatile_enough = atr_14[i] > 0.5 * atr_50[i]
        
        if position == 0 and volatile_enough:
            # Long: price below VWAP AND above 1w EMA200 (mean reversion in uptrend)
            if close[i] < vwap[i] and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP AND below 1w EMA200 (mean reversion in downtrend)
            elif close[i] > vwap[i] and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP OR crosses 1w EMA200
            exit_signal = False
            if position == 1:
                # Exit long when price >= VWAP OR < 1w EMA200
                if close[i] >= vwap[i] or close[i] < ema_200_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price <= VWAP OR > 1w EMA200
                if close[i] <= vwap[i] or close[i] > ema_200_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_VWAP_Deviation_1wEMA200_Trend_ATRRegime"
timeframe = "6h"
leverage = 1.0