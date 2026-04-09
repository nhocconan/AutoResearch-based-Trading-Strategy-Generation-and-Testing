#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d EMA(50) trend + volume confirmation
# Camarilla levels provide precise intraday support/resistance from prior day
# 1d EMA(50) filters for higher timeframe trend direction
# Volume ensures breakout authenticity; discrete sizing 0.25 controls drawdown
# Works in bull/bear: trend filter adapts, Camarilla breakouts work in both directions
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing

name = "4h_1d_camarilla_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for 1d bar close)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate prior day's Camarilla levels (using 1d OHLC)
    # Camarilla levels based on previous day's range
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:  # Need at least one prior day
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Get prior day's OHLC (1d data)
            prev_day_idx = i // 16  # Approximate: 16x 4h bars per day
            if prev_day_idx < len(df_1d):
                prev_high = df_1d['high'].iloc[prev_day_idx]
                prev_low = df_1d['low'].iloc[prev_day_idx]
                prev_close = df_1d['close'].iloc[prev_day_idx]
                range_val = prev_high - prev_low
                
                # Camarilla levels
                camarilla_h3[i] = prev_close + range_val * 1.1 / 4
                camarilla_l3[i] = prev_close - range_val * 1.1 / 4
                camarilla_h4[i] = prev_close + range_val * 1.1 / 2
                camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
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
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 1d EMA (trend change)
            if close[i] < camarilla_l3[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 1d EMA (trend change)
            if close[i] > camarilla_h3[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + 1d EMA filter
            if volume_confirmed:
                # Long entry: price > Camarilla H3 AND price > 1d EMA (bullish alignment)
                if close[i] > camarilla_h3[i] and close[i] > ema_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L3 AND price < 1d EMA (bearish alignment)
                elif close[i] < camarilla_l3[i] and close[i] < ema_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals