#!/usr/bin/env python3
"""
1h_Pivot_R1S1_R2S2_Breakout_Volume_Trend
Hypothesis: Price breaks above/below daily Camarilla pivot resistance/support levels (R1,S1,R2,S2) with volume spike and 4h EMA trend filter.
Uses daily pivots for structure, volume confirmation for momentum, and 4h EMA for trend direction to avoid counter-trend trades.
Targets 15-37 trades/year (60-150 total over 4 years) on 1h timeframe by using daily pivots for direction and 1h only for entry timing.
Works in bull/bear markets: buys strength in uptrends, sells weakness in downtrends.
"""

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
    
    # Get daily OHLC for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous day's OHLC
    # Camarilla: Close ± (High-Low) * 1.1/12, * 1.1/6, * 1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data only
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    r2 = prev_close + range_hl * 1.1 / 6
    s2 = prev_close - range_hl * 1.1 / 6
    
    # Align pivots to 1h timeframe (wait for daily close)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    r2_1h = align_htf_to_ltf(prices, df_1d, r2)
    s2_1h = align_htf_to_ltf(prices, df_1d, s2)
    
    # 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume spike: >1.8x 24-period average (reduced from 2.0 to increase frequency slightly)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 24)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(r2_1h[i]) or np.isnan(s2_1h[i]) or
            np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_1h[i]
        s1 = s1_1h[i]
        r2 = r2_1h[i]
        s2 = s2_1h[i]
        ema4h = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1/R2 with volume spike and 4h uptrend
            if price > r1 and vol_spike and price > ema4h:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1/S2 with volume spike and 4h downtrend
            elif price < s1 and vol_spike and price < ema4h:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price closes below 4h EMA OR breaks below S1 (reversal)
            if price < ema4h:
                signals[i] = 0.0
                position = 0
            elif price < s1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price closes above 4h EMA OR breaks above R1 (reversal)
            if price > ema4h:
                signals[i] = 0.0
                position = 0
            elif price > r1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Pivot_R1S1_R2S2_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0