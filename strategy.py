# 12h Camarilla Pivot Reversal with Volume and 1W Trend Filter
# Hypothesis: Price reverses at Camarilla pivot levels (H3/L3) on 12h timeframe with volume confirmation and weekly trend alignment.
# Works in bull/bear markets by fading extremes with institutional volume and higher timeframe trend filter.
# Target: 20-40 trades/year on 12h timeframe with low turnover to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Camarilla pivot levels from previous day
    # Using daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align daily data to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels
    # H3 = Close + (High - Low) * 1.1 / 4
    # L3 = Close - (High - Low) * 1.1 / 4
    camarilla_h3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    camarilla_l3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 1w trend direction
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Reversal conditions at Camarilla levels
        # Long when price touches L3 in uptrend with volume
        long_signal = (low[i] <= camarilla_l3[i]) and vol_filter and is_uptrend
        # Short when price touches H3 in downtrend with volume
        short_signal = (high[i] >= camarilla_h3[i]) and vol_filter and is_downtrend
        
        # Exit conditions: opposite Camarilla level touch or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on touch of H3 or trend turns down
            exit_long = (high[i] >= camarilla_h3[i]) or (not is_uptrend)
        elif position == -1:
            # Exit short on touch of L3 or trend turns up
            exit_short = (low[i] <= camarilla_l3[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals