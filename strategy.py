# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Breakouts at R1/S1 levels (minor sentiment extremes) with 12h trend filter and volume confirmation.
# R1/S1 provides tighter stops and more frequent entries than R3/S3. 12h trend ensures alignment with medium-term bias.
# Volume filters low-liquidity breakouts. Designed for 20-50 trades/year in both bull and bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R1, S1)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12  # R1 = C + 1.1*(H-L)/12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12  # S1 = C - 1.1*(H-L)/12
    
    # Trend filter: 12h EMA34
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current 4h volume > 1.3 * 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Align all to 4h (primary timeframe)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema34_12h_4h = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 30)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(ema34_12h_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_4h[i]
        s1_val = s1_4h[i]
        trend = ema34_12h_4h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below trend
            elif close[i] < s1_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (mean reversion to center)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 (mean reversion to center)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals