#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and session (08-20 UTC)
# Uses Camarilla pivot levels (R3/S3) from prior 4h bar for structure-based breakouts
# 4h EMA50 confirms trend direction to avoid counter-trend trades in ranging markets
# Session filter (08-20 UTC) reduces noise during low-liquidity Asian hours
# Discrete position sizing (0.20) minimizes fee drag while allowing meaningful exposure
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide mathematical support/resistance, EMA adds trend filter

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate Camarilla levels from prior 4h bar (HLC of completed 4h bar)
    # Typical price = (High + Low + Close) / 3
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    # Camarilla width = (High - Low) * 1.1 / 8
    camarilla_width = (df_4h['high'] - df_4h['low']) * 1.1 / 8.0
    # R3 = Typical + 4 * width, S3 = Typical - 4 * width
    camarilla_r3 = typical_price_4h + 4.0 * camarilla_width
    camarilla_s3 = typical_price_4h - 4.0 * camarilla_width
    
    # Align Camarilla levels to 1h timeframe (use completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for 4h EMA50 calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Camarilla breakout long: price > R3 level
            # Camarilla breakout short: price < S3 level
            breakout_long = close[i] > camarilla_r3_aligned[i]
            breakout_short = close[i] < camarilla_s3_aligned[i]
            
            # 4h EMA50 trend filter: price > EMA for longs, price < EMA for shorts
            ema_long = close[i] > ema_4h_aligned[i]
            ema_short = close[i] < ema_4h_aligned[i]
            
            if breakout_long and ema_long:
                signals[i] = 0.20
                position = 1
            elif breakout_short and ema_short:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 level (reversal) or trend change
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 level (reversal) or trend change
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals