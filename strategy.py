#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d EMA34 trend filter.
# Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) identify key intraday support/resistance.
# Trade breakouts beyond R4/S4 in direction of daily trend (EMA34) with volume confirmation.
# Works in bull/bear markets: avoids false breakouts in ranges, captures true momentum moves.
# Target: 15-35 trades/year per symbol.
name = "6h_Camarilla_R4_S4_Breakout_Volume_EMA34"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Calculate EMA34 on daily for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Align 1d data to 6h
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align 12h volume to 6h (use last known 12h volume)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    # Volume confirmation: current volume > 1.5x 12-period average of 12h volume
    vol_ma_12 = pd.Series(volume_12h_aligned).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 12)  # Ensure EMA34 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_ma = vol_ma_12[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout beyond R4/S4 in direction of daily trend
            if price > r4 and price > ema_34_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif price < s4 and price < ema_34_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to EMA34 or breaks below S4 (failed breakout)
            if price < ema_34_val or price < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to EMA34 or breaks above R4 (failed breakout)
            if price > ema_34_val or price > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals