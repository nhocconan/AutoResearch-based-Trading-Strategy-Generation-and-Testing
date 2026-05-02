#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h HTF for EMA50 to capture intermediate trend and reduce false breakouts.
# Camarilla H3/L3 from 4h provides proven intraday reversal/continuation levels.
# Volume confirmation at 1.8x average ensures strong participation while limiting trades (~15-37/year target).
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete sizing 0.20 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.

name = "1h_Camarilla_H3_L3_Breakout_4hEMA50_Volume"
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
    
    # Calculate Camarilla levels H3 and L3 from 4h timeframe (using prior completed 4h bar)
    # For 1h timeframe, we need to use the prior completed 4h bar's OHLC
    if len(prices) < 4:
        return np.zeros(n)
    
    # Get prior completed 4h bar's OHLC (shift by 4 for 1h timeframe: 4h/1h=4)
    prev_high_4h = prices['high'].shift(4).values
    prev_low_4h = prices['low'].shift(4).values
    prev_close_4h = prices['close'].shift(4).values
    
    # Camarilla H3 and L3 levels (proven breakout/continuation levels)
    camarilla_h3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    camarilla_l3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # 4h EMA50 for trend filter (intermediate trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 1.8x 20-period average (balanced threshold)
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
        
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H3 AND price > 4h EMA50 AND volume spike
            if (close[i] > camarilla_h3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below L3 AND price < 4h EMA50 AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L3 OR price < 4h EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above H3 OR price > 4h EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals