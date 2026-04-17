# 6h Weekly Pivot Breakout with Volume Confirmation
# Hypothesis: Weekly pivots act as strong support/resistance levels. Breakouts above weekly R1/R2 or below S1/S2 with volume confirmation capture institutional flow.
# Works in bull/bear: In bull markets, breaks above R1/R2 continue upward; in bear markets, breaks below S1/S2 continue downward.
# Uses 6h timeframe for balance of signal frequency and noise reduction.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    # Using prior week's data to avoid look-ahead
    weekly_high = df_1w['high'].shift(1)
    weekly_low = df_1w['low'].shift(1)
    weekly_close = df_1w['close'].shift(1)
    
    # Calculate pivot levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align to 6h timeframe (waits for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # Volume confirmation: 6h volume > 1.5x 24-period EMA of volume
    volume_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # warmup for weekly data and volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_ema_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ema = volume_ema_24[i]
        
        if position == 0:
            # Long: break above R1 or R2 with volume confirmation
            if (price > r1_aligned[i] or price > r2_aligned[i]) and vol > 1.5 * vol_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 or S2 with volume confirmation
            elif (price < s1_aligned[i] or price < s2_aligned[i]) and vol > 1.5 * vol_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 (invalidates bullish breakout)
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 (invalidates bearish breakout)
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0