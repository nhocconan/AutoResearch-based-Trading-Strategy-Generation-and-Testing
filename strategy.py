#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and Daily Trend Filter
Hypothesis: On the 12h timeframe, price breaking above/below Camarilla pivot levels (R1/S1) 
from the previous daily session, confirmed by volume (>1.5x average) and aligned with 
daily trend (price above/below daily EMA34), captures high-probability moves in both 
bull and bear markets. The 12h timeframe reduces trade frequency to minimize fee drag, 
while the daily trend filter ensures trades are taken in the direction of the higher-timeframe 
trend, improving win rate. Target: 15-30 trades/year to stay well within fee limits.
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
    
    # --- 1d data for Camarilla pivots and EMA34 (loaded ONCE) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need at least 34 for EMA + 1 for previous day
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each day using previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from the *previous* day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use only previous day's data
    camarilla_width = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Align R1/S1 to 12h timeframe (waits for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter (only needs completed 1d candle)
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- Volume confirmation: volume > 1.5x 30-period average ---
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 30 for volume MA and alignment is handled by align_htf_to_ltf
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1, above daily EMA34, with volume
            if price > r1_val and price > ema_34_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below daily EMA34, with volume
            elif price < s1_val and price < ema_34_val and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to daily EMA34 or breaks below S1 (reversal)
            if price < ema_34_val or price < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to daily EMA34 or breaks above R1 (reversal)
            if price > ema_34_val or price > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_DailyTrend"
timeframe = "12h"
leverage = 1.0