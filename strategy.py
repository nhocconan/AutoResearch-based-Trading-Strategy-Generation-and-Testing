#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 1d HTF for EMA34 to capture longer-term trend and reduce false breakouts.
# Camarilla H4/L4 from 1d provides proven multi-day reversal/continuation levels.
# Volume confirmation at 2.0x average ensures strong participation while limiting trades (~15-37/year target).
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete sizing 0.25 to balance opportunity and fee drag. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "12h_Camarilla_H4_L4_Breakout_1dEMA34_Volume"
timeframe = "12h"
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
    
    # Calculate Camarilla levels H4 and L4 from 1d timeframe (using prior completed 1d bar)
    # Since we're on 12h timeframe, we need to use the prior completed 1d bar's OHLC
    # We'll load 1d data and shift by 1 to get prior completed bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Align the prior completed 1d OHLC to 12h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Camarilla H4 and L4 levels (proven breakout/continuation levels)
    camarilla_h4 = prev_close_1d_aligned + (prev_high_1d_aligned - prev_low_1d_aligned) * 1.1 / 2
    camarilla_l4 = prev_close_1d_aligned - (prev_high_1d_aligned - prev_low_1d_aligned) * 1.1 / 2
    
    # 1d EMA34 for trend filter (longer-term trend)
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H4 AND price > 1d EMA34 AND volume spike
            if (close[i] > camarilla_h4[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 AND price < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_l4[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L4 OR price < 1d EMA34
            if close[i] < camarilla_l4[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above H4 OR price > 1d EMA34
            if close[i] > camarilla_h4[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals