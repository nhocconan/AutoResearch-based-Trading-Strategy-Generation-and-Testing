#!/usr/bin/env python3
name = "12h_Pivots_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot levels (resistance/support)
    H4 = daily_close + 1.5 * (daily_high - daily_low)
    L4 = daily_close - 1.5 * (daily_high - daily_low)
    H3 = daily_close + 1.1 * (daily_high - daily_low)
    L3 = daily_close - 1.1 * (daily_high - daily_low)
    H2 = daily_close + 0.5 * (daily_high - daily_low)
    L2 = daily_close - 0.5 * (daily_high - daily_low)
    H1 = daily_close + 0.25 * (daily_high - daily_low)
    L1 = daily_close - 0.25 * (daily_high - daily_low)
    
    # Align pivot levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(H2_aligned[i]) or np.isnan(L2_aligned[i]) or
            np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above H1 with volume in uptrend
            if close[i] > H1_aligned[i] and vol_condition and ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L1 with volume in downtrend
            elif close[i] < L1_aligned[i] and vol_condition and ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to L2 or trend reverses
            if close[i] < L2_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to H2 or trend reverses
            if close[i] > H2_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla pivot breakouts with daily trend filter and volume confirmation
# - Uses Camarilla levels (H1/L1 for entry, H2/L2 for exit) from daily timeframe
# - Enters long when price breaks above H1 with volume spike in daily uptrend
# - Enters short when price breaks below L1 with volume spike in daily downtrend
# - Exits when price returns to H2/L2 or daily trend reverses
# - Volume confirmation (2x average) reduces false breakouts
# - Designed for 12h timeframe to target 50-150 total trades over 4 years
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Pivot levels provide institutional support/resistance with statistical edge
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Position size 0.25 balances return potential with drawdown control
# - Aims for ~25-35 trades per year to stay within fee-efficient limits
# - Avoids overtrading by requiring multiple confluence factors (level + volume + trend)