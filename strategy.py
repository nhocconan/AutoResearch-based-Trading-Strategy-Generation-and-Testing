#!/usr/bin/env python3
"""
1h Momentum Reversal with 4h Trend Filter and Volume Confirmation
Hypothesis: In BTC/ETH, short-term mean reversion on 1h during strong 4h trends captures 
pullbacks in trending markets. Uses 4h EMA for trend direction, 1h RSI for entry timing, 
and volume confirmation to filter low-quality signals. Designed to work in both bull 
and bear markets by trading with the higher-timeframe trend. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA for trend direction (21-period)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI for mean reversion (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for RSI and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: uptrend + oversold RSI + volume
            if price > ema_trend and rsi_val < 30 and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + overbought RSI + volume
            elif price < ema_trend and rsi_val > 70 and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long if RSI reverts or trend breaks
            if rsi_val > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if RSI reverts or trend breaks
            if rsi_val < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Momentum_Reversal_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0