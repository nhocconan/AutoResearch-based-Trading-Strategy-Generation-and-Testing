#/usr/bin/env python3
# 12h_camilla_pivot_breakout_volume_v2
# Hypothesis: Camarilla pivot levels on daily timeframe with volume confirmation and 1w trend filter for 12h timeframe.
# Long when price breaks above daily R4 level with volume > 1.3x average and weekly close above weekly EMA20.
# Short when price breaks below daily S4 level with volume > 1.3x average and weekly close below weekly EMA20.
# Exit when price returns to daily pivot point or volume drops below average.
# Uses actual daily and weekly data from Binance (no resampling).
# Target: 15-35 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    PP = (high_1d + low_1d + close_1d) / 3
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: 1.3x 24-period average (2 days of 12h data)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.3 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(24, 1) + 1  # Volume MA needs 24 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price back to pivot point or volume drops below average
            if close[i] <= PP_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price back to pivot point or volume drops below average
            if close[i] >= PP_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above R4 with volume surge and weekly uptrend
            if (close[i] > R4_aligned[i] and vol_surge[i] and 
                close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below S4 with volume surge and weekly downtrend
            elif (close[i] < S4_aligned[i] and vol_surge[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals