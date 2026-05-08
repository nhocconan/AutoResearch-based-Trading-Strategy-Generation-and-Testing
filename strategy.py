#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R1_S1_Breakout_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels for current week (calculated from previous week's OHLC)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous week's data, so we shift by 1
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    # First week will have invalid data (rolled from last), handle with nans
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    rng = prev_high - prev_low
    R1 = prev_close + rng * 1.1 / 12
    S1 = prev_close - rng * 1.1 / 12
    
    # Align weekly R1/S1 to daily timeframe (only use after weekly bar closes)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Weekly trend filter: EMA34 (only use after weekly bar closes)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily volume spike: volume > 1.5 * 20-day average (using only past data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND weekly trend up (close > EMA34) AND volume spike
            long_cond = (close[i] > R1_aligned[i]) and (close[i] > ema34_aligned[i]) and vol_spike[i]
            
            # Short: price breaks below S1 AND weekly trend down (close < EMA34) AND volume spike
            short_cond = (close[i] < S1_aligned[i]) and (close[i] < ema34_aligned[i]) and vol_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (reversal) OR weekly trend turns down
            if (close[i] < S1_aligned[i]) or (close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 (reversal) OR weekly trend turns up
            if (close[i] > R1_aligned[i]) or (close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla R1/S1 levels act as strong support/resistance. 
# Breakouts with volume confirmation and weekly trend filter (EMA34) capture institutional moves.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Volume spike ensures breakout validity. Weekly EMA34 filters counter-trend breakouts.
# Target: 20-60 trades over 4 years = 5-15/year to minimize fee decay and avoid overtrading.