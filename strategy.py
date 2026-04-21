#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Reversal_WeeklyTrend_v1
Hypothesis: Daily Camarilla pivot reversals with weekly trend filter. In weekly uptrend, buy at S1/S2/S3 support; in weekly downtrend, sell at R1/R2/R3 resistance. Uses volume confirmation (>1.5x 20-day average) and ATR stop (2.5x) to reduce false breaks. Designed for 1d timeframe to capture swing reversals in both bull and bear markets by aligning with higher timeframe weekly trend. Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily ATR (20-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    # === Volume confirmation (1.5x 20-day MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily Camarilla pivot levels (R1,R2,R3,S1,S2,S3) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + range_val * 1.1 / 12.0
    r2 = pivot + range_val * 1.1 / 6.0
    r3 = pivot + range_val * 1.1 / 4.0
    s1 = pivot - range_val * 1.1 / 12.0
    s2 = pivot - range_val * 1.1 / 6.0
    s3 = pivot - range_val * 1.1 / 4.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        r1_val, r2_val, r3_val = r1[i], r2[i], r3[i]
        s1_val, s2_val, s3_val = s1[i], s2[i], s3[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Weekly uptrend: look for long reversals at support
            if ema_34_1w_val > 0:  # weekly uptrend
                long_condition = volume_confirm and (
                    (price <= s1_val and low[i] < s1_val) or  # price touches/goes below S1
                    (price <= s2_val and low[i] < s2_val) or  # price touches/goes below S2
                    (price <= s3_val and low[i] < s3_val)     # price touches/goes below S3
                )
                # Enter long at close if conditions met
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
            # Weekly downtrend: look for short reversals at resistance
            else:  # weekly downtrend
                short_condition = volume_confirm and (
                    (price >= r1_val and high[i] > r1_val) or  # price touches/goes above R1
                    (price >= r2_val and high[i] > r2_val) or  # price touches/goes above R2
                    (price >= r3_val and high[i] > r3_val)     # price touches/goes above R3
                )
                # Enter short at close if conditions met
                if short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (weekly trend changes)
                elif (ema_34_1w_val <= 0 and position == 1) or (ema_34_1w_val > 0 and position == -1):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (weekly trend changes)
                elif (ema_34_1w_val > 0 and position == -1) or (ema_34_1w_val <= 0 and position == 1):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Reversal_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0