#!/usr/bin/env python3
"""
6h_1d_Volume_Weighted_RSI_Pullback_v1
Hypothesis: On 6h timeframe, buy pullbacks to VWAP when RSI(14) is oversold (<30) with volume confirmation,
and sell rallies to VWAP when RSI is overbought (>70). Uses 1d trend filter (price > 1d EMA50 for longs,
price < 1d EMA50 for shorts) to avoid counter-trend trades. Designed for 15-35 trades/year by requiring
multiple confluence factors: VWAP proximity, RSI extreme, volume spike, and trend alignment.
Works in bull markets via long pullbacks and in bear markets via short rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Volume_Weighted_RSI_Pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data ONCE for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # VWAP proximity: within 0.5% of VWAP
        vwap_dist_pct = abs(close[i] - vwap[i]) / vwap[i] * 100
        near_vwap = vwap_dist_pct <= 0.5
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume spike: current volume > 1.8x average
        volume_spike = volume[i] > vol_ma[i] * 1.8
        
        # Trend filter from 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = near_vwap and rsi_oversold and volume_spike and above_ema
        short_entry = near_vwap and rsi_overbought and volume_spike and below_ema
        
        # Exit conditions: RSI returns to neutral zone (40-60) or VWAP breach
        long_exit = rsi[i] >= 40 or close[i] < vwap[i]
        short_exit = rsi[i] <= 60 or close[i] > vwap[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals