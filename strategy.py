#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1h
# - Trend filter: 4h EMA(50) slope > 0 for longs, < 0 for shorts (only trade with trend)
# - Session filter: Trade only between 08:00-20:00 UTC to avoid low-liquidity hours
# - Works in bull/bear: Trend filter ensures we only take trades in direction of 4h momentum
# - Position size: 0.20 discrete level to minimize fee churn
# - Target: 15-35 trades/year (60-140 total over 4 years) per 1h strategy guidelines

name = "1h_4h_camarilla_ema_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h EMA(50) and its slope for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = ema_50 - np.roll(ema_50, 1)  # current - previous
    ema_slope[0] = 0
    ema_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_slope)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 1h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3 = C +/- (H-L)*1.1/2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup for EMA
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reversion to Camarilla H3 level or trend change
            if prices['close'].iloc[i] <= camarilla_h3_aligned[i] or ema_slope_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reversion to Camarilla L3 level or trend change
            if prices['close'].iloc[i] >= camarilla_l3_aligned[i] or ema_slope_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with trend confirmation
            # Long: price breaks above H3 with upward 4h EMA slope
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                ema_slope_aligned[i] > 0):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.20
            # Short: price breaks below L3 with downward 4h EMA slope
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  ema_slope_aligned[i] < 0):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.20
    
    return signals