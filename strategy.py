#!/usr/bin/env python3
"""
1h_vwap_rsi_divergence_4h1d_trend_v1
Hypothesis: On 1h timeframe, enter long when price > VWAP, RSI < 35 (oversold), and 4h trend up (price > 4h EMA50). Enter short when price < VWAP, RSI > 65 (overbought), and 4h trend down (price < 4h EMA50). Use daily volume filter (volume > 1.5x 20-day average) to avoid low-volume noise. Target: 15-35 trades/year by combining mean reversion (RSI extremes) with trend filter and volume confirmation. Works in bull/bear: trend filter prevents counter-trend trades, VWAP/RSI capture mean reversion within trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_rsi_divergence_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = cum_pv / (cum_vol + 1e-10)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily volume filter (volume > 1.5x 20-day average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x daily average
        vol_ok = volume[i] > (vol_ma_1d_aligned[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 65 (overbought) or trend changes or price < VWAP
            if rsi[i] > 65 or close[i] < ema_50_4h_aligned[i] or close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 35 (oversold) or trend changes or price > VWAP
            if rsi[i] < 35 or close[i] > ema_50_4h_aligned[i] or close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price > VWAP, RSI < 35 (oversold), 4h trend up
                if (close[i] > vwap[i] and 
                    rsi[i] < 35 and 
                    close[i] > ema_50_4h_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: price < VWAP, RSI > 65 (overbought), 4h trend down
                elif (close[i] < vwap[i] and 
                      rsi[i] > 65 and 
                      close[i] < ema_50_4h_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals