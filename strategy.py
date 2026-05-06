#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
# with 1d EMA34 trend filter and volume confirmation. 
# Long when price breaks above R4 with EMA34 rising and volume spike, or mean reverts from S3 with EMA34 rising and volume spike. 
# Short when price breaks below S4 with EMA34 falling and volume spike, or mean reverts from R3 with EMA34 falling and volume spike. 
# Exit when price reaches the 12h Camarilla pivot point (PP). 
# Uses discrete sizing 0.25 to balance profit potential and drawdown control. 
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe. 
# 12h Camarilla provides strong intraday structure, 1d EMA34 ensures we trade with the daily trend, volume confirmation filters low-conviction moves. 
# Works in both bull (breakout continuations) and bear (mean reverts from extreme levels) markets.

name = "6h_12hCamarilla_R3S3R4S4_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for pivot calculation (using previous bar)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar)
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # R3 = PP + (H - L) * 1.1/4
    # S3 = PP - (H - L) * 1.1/4
    # S4 = PP - (H - L) * 1.1/2
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r4_12h = pp_12h + (high_12h - low_12h) * 1.1 / 2.0
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 4.0
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 4.0
    s4_12h = pp_12h - (high_12h - low_12h) * 1.1 / 2.0
    
    # Align 12h Camarilla levels to 6h timeframe (wait for completed 12h bar)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA34 and volume avg
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with EMA34 rising and volume spike
            if (close[i] > r4_12h_aligned[i] and close[i-1] <= r4_12h_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Long mean reversion: price crosses above S3 with EMA34 rising and volume spike
            elif (close[i] > s3_12h_aligned[i] and close[i-1] <= s3_12h_aligned[i-1] and 
                  ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with EMA34 falling and volume spike
            elif (close[i] < s4_12h_aligned[i] and close[i-1] >= s4_12h_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            # Short mean reversion: price crosses below R3 with EMA34 falling and volume spike
            elif (close[i] < r3_12h_aligned[i] and close[i-1] >= r3_12h_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches the 12h Camarilla pivot point (PP)
            if close[i] >= pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches the 12h Camarilla pivot point (PP)
            if close[i] <= pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals