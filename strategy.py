#!/usr/bin/env python3
"""
1h Volume-Weighted RSI with 4h Trend and 1d Momentum Filter
Hypothesis: Combines RSI mean reversion with volume confirmation for entry timing,
filtered by 4h EMA trend and 1d momentum to avoid counter-trend trades.
Targets 15-35 trades/year on 1h timeframe by using strict multi-timeframe alignment.
Works in bull markets via trend alignment and bear markets via mean reversion within trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_rsi_4h_trend_1d_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # RSI(14) on close prices
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = df_4h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI(14) for momentum filter
    df_1d = get_htf_data(prices, '1d')
    delta_1d = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: current volume > 1.5x 20-period VWAP-weighted average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) OR price < VWAP (mean reversion failure)
            if (rsi[i] > 60 or close[i] < vwap[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) OR price > VWAP (mean reversion failure)
            if (rsi[i] < 40 or close[i] > vwap[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI < 30 (oversold), price > VWAP, uptrend (4h EMA), bullish momentum (1d RSI > 50)
            if (rsi[i] < 30 and close[i] > vwap[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                rsi_1d_aligned[i] > 50 and vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI > 70 (overbought), price < VWAP, downtrend (4h EMA), bearish momentum (1d RSI < 50)
            elif (rsi[i] > 70 and close[i] < vwap[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals