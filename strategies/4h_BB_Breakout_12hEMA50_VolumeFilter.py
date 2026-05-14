#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 12h trend filter and volume confirmation.
# Long when price breaks above upper BB with bullish trend and volume spike.
# Short when price breaks below lower BB with bearish trend and volume spike.
# Exit when price returns to middle BB (mean reversion).
# Uses 12h timeframe for trend filter to reduce noise and improve win rate.
# Target: 20-40 trades/year to minimize fee drag while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12-hour EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Bollinger Bands on 4h timeframe (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std_dev * bb_std)
    lower_bb = sma_bb - (bb_std_dev * bb_std)
    middle_bb = sma_bb
    
    # Volume filter: volume > 1.5x 20-period average (balanced to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper BB, above 12h EMA50, volume spike
        if (close[i] > upper_bb[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower BB, below 12h EMA50, volume spike
        elif (close[i] < lower_bb[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle BB (mean reversion)
        elif position == 1 and close[i] < middle_bb[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > middle_bb[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_BB_Breakout_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0