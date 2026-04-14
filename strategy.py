# 12h Camarilla Pivot Breakout with Volume Confirmation
# Hypothesis: Camarilla pivot levels from 1d provide robust support/resistance levels
# Breaking above/below these levels with volume confirmation indicates institutional interest
# Works in bull/bear markets as it follows institutional flow rather than trend
# Uses 12h timeframe to reduce trade frequency and fee drag
# Target: 20-40 trades per year per symbol

#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels using prior day's OHLC
    # Formula: P = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1/2, L3 = C - (H - L) * 1.1/2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point
    p = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    h4 = prev_close + (prev_high - prev_low) * 1.1
    l4 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align pivot levels to 12h timeframe
    p_aligned = align_htf_to_ltf(prices, df_1d, p)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: volume > 1.8x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 24  # for volume calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(p_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above H4 with volume confirmation
            if price > h4_aligned[i] and vol > 1.8 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L4 with volume confirmation
            elif price < l4_aligned[i] and vol > 1.8 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below H3 (profit taking or reversal)
            if price < h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above L3 (profit taking or reversal)
            if price > l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0