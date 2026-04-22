#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 12h EMA filter and volume confirmation.
# Uses daily Camarilla levels (R1/S1, R2/S2) to identify mean reversion zones.
# Enters long at S1/S2 bounce in 12h uptrend, short at R1/R2 rejection in 12h downtrend.
# Requires volume spike for confirmation. Designed for 6h timeframe to capture
# intraday reversals with multi-day trend context. Target: 20-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using prior day's data
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # S1 = C - (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # S2 = C - (Range * 1.1 / 6)
    range_1d = high_1d - low_1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    
    # Load 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Align indicators to 6-hour timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at S1/S2 + 12h uptrend + volume spike
            if ((close[i] <= s1_1d_aligned[i] * 1.005 or close[i] <= s2_1d_aligned[i] * 1.005) and
                close[i] > ema_34_12h_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at R1/R2 + 12h downtrend + volume spike
            elif ((close[i] >= r1_1d_aligned[i] * 0.995 or close[i] >= r2_1d_aligned[i] * 0.995) and
                  close[i] < ema_34_12h_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite level or trend changes
            if position == 1:
                if (close[i] >= r1_1d_aligned[i] * 0.995 or close[i] <= s2_1d_aligned[i] * 0.995 or
                    close[i] < ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] <= s1_1d_aligned[i] * 1.005 or close[i] >= r2_1d_aligned[i] * 1.005 or
                    close[i] > ema_34_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_R2_S2_12hEMA34_Volume_Spike"
timeframe = "6h"
leverage = 1.0