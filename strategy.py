#!/usr/bin/env python3
"""
4h_12h_1d_Camarilla_Breakout_Plus_v2
Hypothesis: On 4h timeframe, trade Camarilla breakouts with 12h trend filter and 1d volatility regime.
Long when price breaks above H4 with 12h uptrend and low volatility regime; short when breaks below L4 with 12h downtrend and low volatility.
Exit at opposite H3/L3 levels. Uses volume confirmation (1.5x average) to avoid false breakouts.
Designed for low trade frequency (20-40/year) by requiring Camarilla level confluence, trend alignment, and volatility filter.
Works in bull/bear via 12h trend filter and mean-reversion exit at Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_Camarilla_Breakout_Plus_v2"
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
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    range_1d = high_1d - low_1d
    
    # H4, L4, H3, L3 levels
    h4 = close_1d + range_1d * 1.1 / 2
    l4 = close_1d - range_1d * 1.1 / 2
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    
    # === 12H EMA(20) FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 20:
        ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_20_12h = np.full_like(close_12h, np.nan)
    
    # === 1D VOLATILITY REGIME (ATR RATIO) ===
    # Calculate daily ATR(10) and its 20-period moving average
    if len(high_1d) >= 10:
        tr1 = high_1d[1:] - low_1d[:-1]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_10 = np.zeros_like(close_1d)
        atr_10[9:] = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
        atr_ma_20 = np.zeros_like(close_1d)
        atr_ma_20[19:] = pd.Series(atr_10).rolling(window=20, min_periods=20).mean().values
        # Avoid division by zero
        atr_ratio = np.zeros_like(close_1d)
        mask = atr_ma_20 > 0
        atr_ratio[mask] = atr_10[mask] / atr_ma_20[mask]
    else:
        atr_ratio = np.zeros_like(close_1d)
    
    # Align data to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume average (20-period for 4h = ~3.3 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below 12h EMA(20)
        price_above_ema = close[i] > ema_20_12h_aligned[i]
        price_below_ema = close[i] < ema_20_12h_aligned[i]
        
        # Volatility regime: low volatility (ATR ratio < 0.8) for better breakout quality
        low_vol = atr_ratio_aligned[i] < 0.8
        
        # Entry conditions
        long_setup = (close[i] > h4_aligned[i]) and vol_confirm and price_above_ema and low_vol
        short_setup = (close[i] < l4_aligned[i]) and vol_confirm and price_below_ema and low_vol
        
        # Exit conditions: mean reversion to H3/L3 levels
        exit_long = close[i] < l3_aligned[i]
        exit_short = close[i] > h3_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals