#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Price breaking above/below R1/S1 Camarilla levels on 1h, filtered by 4h EMA trend and volume spike (2x median), with session filter (08-20 UTC). Uses 4h trend for direction, 1h for entry timing. Designed to work in bull (uptrend breaks) and bear (downtrend breaks). Target: 15-35 trades/year.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    volume_1h = prices['volume'].values
    
    # --- 4h Trend Filter: EMA50 ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- 1h Camarilla Levels (based on previous 1h bar) ---
    prev_close = np.roll(close_1h, 1)
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close[0] = close_1h[0]
    prev_high[0] = high_1h[0]
    prev_low[0] = low_1h[0]
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # --- Volume Filter: spike above 2x median of last 20 periods ---
    vol_median = pd.Series(volume_1h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 2.0
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_1h - low_1h)
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Session Filter: 08-20 UTC ---
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                # Check stoploss
                if position == 1 and close_1h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Determine 4h trend
        trend_up = close_1h[i] > ema50_4h_aligned[i]
        trend_down = close_1h[i] < ema50_4h_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_1h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume spike
            if close_1h[i] > camarilla_r1[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 4h uptrend + volume spike
                signals[i] = 0.20
                position = 1
                entry_price = close_1h[i]
            elif close_1h[i] < camarilla_s1[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 4h downtrend + volume spike
                signals[i] = -0.20
                position = -1
                entry_price = close_1h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_1h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S1
                elif close_1h[i] <= camarilla_s1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Stoploss
                if close_1h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R1
                elif close_1h[i] >= camarilla_r1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals