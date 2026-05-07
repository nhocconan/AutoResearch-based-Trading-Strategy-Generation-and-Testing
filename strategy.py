#134944
#!/usr/bin/env python3
name = "1d_1wPivot_R1S1_Breakout_Trend_Filter_v3"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly high/low/close for pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for pivot calculation
    prev_high = np.concatenate([[high_1w[0]], high_1w[:-1]])
    prev_low = np.concatenate([[low_1w[0]], low_1w[:-1]])
    prev_close = np.concatenate([[close_1w[0]], close_1w[:-1]])
    
    # Weekly pivot point and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Weekly R1 and S1 levels (standard formula)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily EMA(34) for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 1.5
            uptrend = close[i] > ema_34[i]
            
            if close[i] > r1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and in downtrend
            elif close[i] < s1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA or volume drops
            if close[i] < ema_34[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA or volume drops
            if close[i] > ema_34[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly R1/S1 breakout on daily chart with EMA34 trend filter and volume confirmation.
# Uses weekly pivot levels as key institutional support/resistance. Breakouts above R1 or below S1
# with volume indicate strong institutional participation. Trend filter ensures trades align with
# weekly direction via daily EMA34. Works in bull (buy R1 breaks in uptrend) and bear (sell S1 breaks in downtrend).
# Position size 0.25 limits risk and keeps trade frequency ~15-25/year. Weekly timeframe reduces noise
# and false breakouts compared to daily pivots. Weekly levels are more significant and less prone to
# whipsaw, making this suitable for both trending and ranging markets. The volume confirmation
# ensures breakouts have conviction, avoiding false signals in low-volume environments.