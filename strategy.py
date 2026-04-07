#!/usr/bin/env python3
"""
1d_kama_rsi_chop_filter_v1
Hypothesis: On daily timeframe, use KAMA to determine trend direction, RSI for momentum confirmation, and Choppiness Index to filter ranging markets. Enter long when KAMA turns bullish, RSI > 50, and CHOP < 40 (trending); enter short when KAMA turns bearish, RSI < 50, and CHOP < 40. Exit when opposite signal occurs. This strategy avoids whipsaws in ranging markets and captures trends in both bull and bear regimes with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for Choppiness Index (trend/ranging filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log15(sum(TR14) / (HH14 - LL14))
    chop_1w = 100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14)
    
    # Align Chop to daily timeframe
    chop_1d = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # KAMA on daily close
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    change = np.concatenate([[np.nan]*10, change])  # align
    
    volatility = np.abs(np.diff(close))  # 1-period change
    volatility = np.concatenate([[np.nan], volatility])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    er = change / vol_sum
    er = np.where(vol_sum == 0, 0, er)  # avoid division by zero
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = np.where(np.isnan(sc), 0.01, sc)  # default to slowest
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI on daily close
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1d[i]) or
            chop_1d[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Trend/trading regime filter: CHOP < 40 = trending market
        trending = chop_1d[i] < 40
        
        if position == 1:  # Long position
            # Exit conditions: KAMA turns bearish OR chop becomes ranging
            exit_long = False
            if i > 0 and kama[i] < kama[i-1]:  # KAMA turning down
                exit_long = True
            elif not trending:  # market becoming ranging
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: KAMA turns bullish OR chop becomes ranging
            exit_short = False
            if i > 0 and kama[i] > kama[i-1]:  # KAMA turning up
                exit_short = True
            elif not trending:  # market becoming ranging
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in trending markets
            if trending:
                # Long entry: KAMA turning up AND RSI > 50
                long_entry = False
                if i > 0 and kama[i] > kama[i-1] and rsi[i] > 50:
                    long_entry = True
                
                # Short entry: KAMA turning down AND RSI < 50
                short_entry = False
                if i > 0 and kama[i] < kama[i-1] and rsi[i] < 50:
                    short_entry = True
                
                if long_entry:
                    position = 1
                    signals[i] = 0.25
                elif short_entry:
                    position = -1
                    signals[i] = -0.25
    
    return signals