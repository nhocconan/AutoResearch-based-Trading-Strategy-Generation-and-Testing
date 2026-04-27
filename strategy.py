#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies the primary trend on daily timeframe. 
Entries occur when price crosses above/below KAMA with RSI confirmation (not overbought/oversold) 
and choppy market filter (Choppiness Index > 61.8 = range, we avoid range; < 38.2 = trending). 
Exits on opposite KAMA cross. Uses 1d timeframe targeting 30-80 trades over 4 years (7-20/year). 
Works in bull markets via trend continuation and bear markets via trend reversals. 
Volume confirmation adds robustness. Uses discrete position sizing (0.25) to minimize fee drag.
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
    
    # Get 1w data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on 1d (ER=10, FAST=2, SLOW=30)
    # Efficiency Ratio = |net change| / sum(|changes|)
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))  # net change over 10 periods
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()  # sum of absolute changes
    er = (change / volatility).replace(0, np.nan)  # avoid division by zero
    # Smoothing constants: fastest = 2/(2+1)=0.6667, slowest = 2/(30+1)=0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate 1w EMA34 for trend filter (only long in uptrend, short in downtrend)
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate RSI(14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high - low
    tr2 = abs(high - close_series.shift(1))
    tr3 = abs(low - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), EMA34 (34), RSI (14), CHOP (14), volume avg (20)
    start_idx = max(10, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(chop_values[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        ema_1w_val = ema_34_1w_aligned[i]
        rsi_val = rsi_values[i]
        chop_val = chop_values[i]
        vol_conf = volume_confirm[i]
        
        # Regime filter: avoid choppy markets (CHOP > 61.8), only trade when trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Determine 1w trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_1w_val
            is_downtrend = close_val < ema_1w_val
            
            # Long conditions: price > KAMA, RSI not overbought (<70), trending regime, volume confirm
            if is_uptrend and (close_val > kama_val) and (rsi_val < 70) and is_trending and vol_conf:
                signals[i] = size
                position = 1
            # Short conditions: price < KAMA, RSI not oversold (>30), trending regime, volume confirm
            elif is_downtrend and (close_val < kama_val) and (rsi_val > 30) and is_trending and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price < KAMA or trend changes to downtrend or choppy market
            exit_condition = (close_val < kama_val) or (close_val < ema_1w_val) or (chop_val > 61.8)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA or trend changes to uptrend or choppy market
            exit_condition = (close_val > kama_val) or (close_val > ema_1w_val) or (chop_val > 61.8)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0