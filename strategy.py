#!/usr/bin/env python3
"""
1d_weekly_pivot_reversion_v3
Hypothesis: On 1d timeframe, use weekly reversal points from prior week's high/low/close for mean reversion. Enter long when price closes below weekly S3 (deep oversold) with RSI < 30 and volume > 1.5x average; enter short when price closes above weekly R3 (deep overbought) with RSI > 70 and volume > 1.5x average. Exit when price reaches opposite weekly H3/L3 level or RSI reverts to 50. Uses weekly structure to avoid noise and focuses on extreme reversals. Targets 10-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_reversion_v3"
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
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    # Calculate weekly reversal points from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ph = df_1w['high'].values  # previous week high
    pl = df_1w['low'].values   # previous week low
    pc = df_1w['close'].values # previous week close
    
    # Calculate weekly reversal levels (similar to Camarilla but using weekly range)
    # S3 = close - 1.1 * (high - low) / 4
    # R3 = close + 1.1 * (high - low) / 4
    weekly_s3 = pc - 1.1 * (ph - pl) / 4
    weekly_r3 = pc + 1.1 * (ph - pl) / 4
    weekly_s2 = pc - 1.1 * (ph - pl) / 6
    weekly_r2 = pc + 1.1 * (ph - pl) / 6
    
    # Align to daily timeframe
    s3_1d = align_htf_to_ltf(prices, df_1w, weekly_s3)
    r3_1d = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s2_1d = align_htf_to_ltf(prices, df_1w, weekly_s2)
    r2_1d = align_htf_to_ltf(prices, df_1w, weekly_r2)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(s3_1d[i]) or np.isnan(r3_1d[i]) or
            np.isnan(s2_1d[i]) or np.isnan(r2_1d[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches S2 level (take profit)
            if close[i] <= s2_1d[i]:
                exit_long = True
            # Exit if RSI returns to 50 (mean reversion complete)
            elif rsi[i] >= 50 and rsi[i-1] < 50:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches R2 level (take profit)
            if close[i] >= r2_1d[i]:
                exit_short = True
            # Exit if RSI returns to 50 (mean reversion complete)
            elif rsi[i] <= 50 and rsi[i-1] > 50:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes below S3 with RSI < 30 and volume confirmation
            long_entry = False
            if (close[i] < s3_1d[i] and rsi[i] < 30 and vol_confirm):
                long_entry = True
            
            # Short entry: price closes above R3 with RSI > 70 and volume confirmation
            short_entry = False
            if (close[i] > r3_1d[i] and rsi[i] > 70 and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals