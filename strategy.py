#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Camarilla pivot levels as structure.
# Long when: price breaks above R1 with volume confirmation and bullish 1d trend
# Short when: price breaks below S1 with volume confirmation and bearish 1d trend
# Exit when price returns to pivot (PP) or opposite S1/R1 level
# Uses Camarilla pivot structure for mean reversion/breakout logic, volume filter to avoid false breaks,
# and 1d trend filter to align with higher timeframe direction. Designed for ~15-25 trades/year per symbol.
name = "6h_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # We need previous day's values, so shift by 1
    pp_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    r1_1d = np.roll(close_1d, 1) + (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 12
    s1_1d = np.roll(close_1d, 1) - (np.roll(high_1d, 1) - np.roll(low_1d, 1)) * 1.1 / 12
    
    # Handle first day (no previous data)
    pp_1d[0] = np.nan
    r1_1d[0] = np.nan
    s1_1d[0] = np.nan
    
    # Align 1d pivot levels to 6h timeframe (only available after 1d bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        pp = pp_aligned[i]
        ema_34 = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and bullish trend (price > EMA34)
            if price > r1 and vol_ratio_val > 1.5 and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and bearish trend (price < EMA34)
            elif price < s1 and vol_ratio_val > 1.5 and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point (PP) or breaks below S1 (reversal)
            if price <= pp or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point (PP) or breaks above R1 (reversal)
            if price >= pp or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals