# 12h Camarilla Pivot Breakout with Volume Confirmation and Trend Filter
# Hypothesis: Camarilla pivot levels act as strong support/resistance in ranging and trending markets.
# Breakouts above R3 or below S3 with volume confirmation and higher timeframe trend filter
# capture institutional moves. Works in bull/bear by filtering breakouts with 1w trend.
# Target: 15-25 trades/year (60-100 total) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Using daily high/low/close from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First value: use current day's values as fallback
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Pivot point and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_val * 1.1 / 2)
    R2 = pivot + (range_val * 1.1 / 4)
    R1 = pivot + (range_val * 1.1 / 6)
    S1 = pivot - (range_val * 1.1 / 6)
    S2 = pivot - (range_val * 1.1 / 4)
    S3 = pivot - (range_val * 1.1 / 2)
    
    # Volume filter: 2.0x 30-period average (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    # 1-week trend filter (Higher Time Frame)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    # EMA 34 on weekly close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        weekly_trend_up = price > ema_34_1w_aligned[i]
        weekly_trend_down = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: break above R3 with volume and weekly uptrend
            if price > R3[i] and vol_ok and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and weekly downtrend
            elif price < S3[i] and vol_ok and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to pivot or weekly trend changes
            if price < pivot[i] or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to pivot or weekly trend changes
            if price > pivot[i] or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0