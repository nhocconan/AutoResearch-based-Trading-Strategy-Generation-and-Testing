#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with volume confirmation and 4h trend filter
# Uses 4h EMA(21) for trend direction and 1d Camarilla levels (H3/L3) for breakout entries
# Only takes breakouts in direction of 4h trend with volume > 1.5x 20-period average
# Position size 0.20 to manage drawdown and enable multiple concurrent positions
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag
# Session filter 08-20 UTC to avoid low-liquidity periods
# Works in both bull/bear: 4h trend filter ensures we trade with higher timeframe momentum

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:  # Need previous day's data
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Camarilla calculation using previous day's range
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_h3[i] = close_1d[i-1] + rng * 1.1 / 4
            camarilla_l3[i] = close_1d[i-1] - rng * 1.1 / 4
            camarilla_h4[i] = close_1d[i-1] + rng * 1.1 / 2
            camarilla_l4[i] = close_1d[i-1] - rng * 1.1 / 2
    
    # Align 1d Camarilla levels to 1h timeframe
    camarilla_h3_1h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_1h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_1h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(camarilla_h3_1h[i]) or 
            np.isnan(camarilla_l3_1h[i]) or 
            np.isnan(avg_volume[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Camarilla L3 OR trend turns unfavorable
            if close[i] < camarilla_l3_1h[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla H3 OR trend turns unfavorable
            if close[i] > camarilla_h3_1h[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation and trend filter
            if volume_confirm:
                # Long breakout: price closes above Camarilla H3 with uptrend
                if close[i] > camarilla_h3_1h[i] and close[i] > ema_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short breakout: price closes below Camarilla L3 with downtrend
                elif close[i] < camarilla_l3_1h[i] and close[i] < ema_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals