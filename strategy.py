#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Uses 4h HTF for EMA34 to capture intermediate trend and reduce false breakouts.
# Camarilla R3/S3 from prior completed 1h bar provides proven intraday support/resistance.
# Volume confirmation at 1.8x average ensures strong participation while limiting trades (~15-37/year target).
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete sizing 0.20 to minimize fee churn. Trend filter ensures trades only with momentum.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.
# Works in bull/bear: trend filter avoids counter-trend entries during strong moves.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from prior completed 1h bar (shift by 1)
    if len(prices) < 2:
        return np.zeros(n)
    
    # Get prior completed 1h bar's high/low/close (shift by 1 for 1h timeframe)
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Camarilla R3/S3 levels from prior completed bar
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 4
    s3 = prev_close - 1.1 * camarilla_range / 4
    
    # 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: 1.8x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 AND price > 4h EMA34 AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 AND price < 4h EMA34 AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Camarilla S3 OR price < 4h EMA34
            if close[i] < s3[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above Camarilla R3 OR price > 4h EMA34
            if close[i] > r3[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals