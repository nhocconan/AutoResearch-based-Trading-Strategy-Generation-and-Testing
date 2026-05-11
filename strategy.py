#%%
#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Price breaking above/below R1/S1 Camarilla levels on 12h, filtered by 1-week EMA200 trend and volume spike (2x median).
Focus on tight entries (R1/S1) with strong long-term trend and volume confirmation to reduce trades and improve win rate.
Works in bull via uptrend breaks, in bear via downtrend breaks. Volume confirms conviction. Target: 12-37 trades/year.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1w Trend Filter: EMA200 ---
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # --- 12h Camarilla Levels (based on previous day) ---
    # Calculate from previous 12h bar (shifted by 1 to avoid lookahead)
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close[0] = close_12h[0]
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    
    # Camarilla R1 and S1 levels (tighter than R3/S3)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # --- Volume Filter: spike above 2x median of last 30 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=30, min_periods=15).median().values
    vol_threshold = vol_median * 2.0
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 100  # for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_12h[i] > ema200_1w_aligned[i]
        trend_down = close_12h[i] < ema200_1w_aligned[i]
        
        # Volume filter: spike above 2x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with volume spike
            if close_12h[i] > camarilla_r1[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 1w uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < camarilla_s1[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 1w downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_12h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S1
                elif close_12h[i] <= camarilla_s1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_12h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R1
                elif close_12h[i] >= camarilla_r1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#%%