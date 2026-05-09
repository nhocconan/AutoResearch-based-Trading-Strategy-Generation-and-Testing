#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout from daily levels with volume confirmation.
# Uses daily Camarilla levels (R3/S3 breakout for continuation, R4/S4 for reversal)
# to capture intraday momentum in both trending and ranging markets.
# Volume filter ensures breakouts have conviction. Designed for 60-100 trades/year.
name = "6h_Camarilla_R3S4_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    S4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align to 6t
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price > R3 with volume and above daily EMA34
            if price > R3_aligned[i] and vol_confirm[i] and price > ema_34_aligned[i]:
                # Strong breakout: reverse at R4
                if price >= R4_aligned[i]:
                    signals[i] = -0.25  # short reversal at R4
                    position = -1
                else:
                    signals[i] = 0.25   # continue long
                    position = 1
            # Short breakdown: price < S3 with volume and below daily EMA34
            elif price < S3_aligned[i] and vol_confirm[i] and price < ema_34_aligned[i]:
                # Strong breakdown: reverse at S4
                if price <= S4_aligned[i]:
                    signals[i] = 0.25   # long reversal at S4
                    position = 1
                else:
                    signals[i] = -0.25  # continue short
                    position = -1
        
        elif position == 1:
            # Exit long: price < R3 or strong reversal at R4
            if price < R3_aligned[i] or price >= R4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > S3 or strong reversal at S4
            if price > S3_aligned[i] or price <= S4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals