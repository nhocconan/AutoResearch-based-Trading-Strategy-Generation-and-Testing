# Hypothesis: 4h price reversal at weekly pivot levels with volume confirmation and trend filter.
# Weekly pivot provides strong support/resistance that works across market regimes.
# Volume spike confirms institutional interest. EMA50 trend filter avoids counter-trend trades.
# Designed for fewer trades (<50/year) to minimize fee drag in both bull and bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for pivot calculation
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points using prior week's HLC (no look-ahead)
    prev_high_w = np.roll(high_w, 1)
    prev_low_w = np.roll(low_w, 1)
    prev_close_w = np.roll(close_w, 1)
    prev_high_w[0] = np.nan
    prev_low_w[0] = np.nan
    prev_close_w[0] = np.nan
    
    pp_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    r1_w = 2 * pp_w - prev_low_w
    s1_w = 2 * pp_w - prev_high_w
    r2_w = pp_w + (prev_high_w - prev_low_w)
    s2_w = pp_w - (prev_high_w - prev_low_w)
    
    # 4h EMA50 for trend filter
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly pivot levels to 4h timeframe
    pp_w_aligned = align_htf_to_ltf(prices, df_w, pp_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    ema50_aligned = align_htf_to_ltf(prices, df_w, ema50)  # align EMA50 to weekly boundary
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(pp_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pp = pp_w_aligned[i]
        r1 = r1_w_aligned[i]
        s1 = s1_w_aligned[i]
        r2 = r2_w_aligned[i]
        s2 = s2_w_aligned[i]
        ema50_val = ema50_aligned[i]
        
        if position == 0:
            # Long: price bounces off S2 with volume surge and above EMA50
            if price <= s2 * 1.002 and price >= s2 * 0.998 and vol > 2.0 * vol_ma and price > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: price rejected at R2 with volume surge and below EMA50
            elif price >= r2 * 0.998 and price <= r2 * 1.002 and vol > 2.0 * vol_ma and price < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price moves back to weekly pivot or opposite extreme
            if position == 1:
                if price >= pp * 0.998 and price <= pp * 1.002:  # reached pivot
                    signals[i] = 0.0
                    position = 0
                elif price >= r1 * 0.998 and price <= r1 * 1.002:  # reached R1 (take profit)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price >= pp * 0.998 and price <= pp * 1.002:  # reached pivot
                    signals[i] = 0.0
                    position = 0
                elif price >= s1 * 0.998 and price <= s1 * 1.002:  # reached S1 (take profit)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WeeklyPivot_S2_R2_Bounce_Volume_EMA50"
timeframe = "4h"
leverage = 1.0