#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation
# Uses 1d EMA50 trend filter to ensure trades align with higher timeframe direction
# Volume confirmation reduces false breakouts
# Camarilla pivot levels provide precise entry/exit points with built-in risk management
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in bull markets via R1 breakouts and in bear via S1 breakdowns
name = "12h_Camarilla_R1S1_Breakout_TrendVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    # R4 = C + (H-L) * 1.500
    # R3 = C + (H-L) * 1.250
    # R2 = C + (H-L) * 1.166
    # R1 = C + (H-L) * 1.083
    # PP = (H+L+C)/3
    # S1 = C - (H-L) * 1.083
    # S2 = C - (H-L) * 1.166
    # S3 = C - (H-L) * 1.250
    # S4 = C - (H-L) * 1.500
    
    camarilla_r1 = typical_price + range_hl * 1.083
    camarilla_s1 = typical_price - range_hl * 1.083
    camarilla_pp = typical_price  # Pivot point
    
    # AlCamarilla levels to 12h timeframe (will only update after 1d bar closes)
    r1_1d = camarilla_r1.values
    s1_1d = camarilla_s1.values
    pp_1d = camarilla_pp.values
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        if position == 0:
            # Long: break above R1 + volume + 1d uptrend (price > EMA50)
            if high[i] > r1_aligned[i-1] and volume_filter and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume + 1d downtrend (price < EMA50)
            elif low[i] < s1_aligned[i-1] and volume_filter and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or ATR-based stop
            if close[i] < s1_aligned[i] or close[i] < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or ATR-based stop
            if close[i] > r1_aligned[i] or close[i] > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals