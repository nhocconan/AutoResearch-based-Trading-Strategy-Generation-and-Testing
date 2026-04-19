#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above R1 (bullish) AND price > 12h EMA34 (uptrend) AND volume > 1.5x daily average volume.
# Short when price breaks below S1 (bearish) AND price < 12h EMA34 (downtrend) AND volume > 1.5x daily average volume.
# Exit when price crosses back through the pivot point (PP).
# Uses Camarilla for precise intraday levels, EMA for trend filter, volume for confirmation.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_Pivot_R1S1_EMA_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels using previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, PP = (H+L+C)/3
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    pp = (prev_high + prev_low + prev_close) / 3
    # Align to 4h timeframe (wait for previous day's close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    
    # Get 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Get daily average volume for confirmation (20-day average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        
        if position == 0:
            # Long entry: break above R1 + uptrend + volume spike
            if price > r1 and price > ema34 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + downtrend + volume spike
            elif price < s1 and price < ema34 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals