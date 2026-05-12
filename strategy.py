#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h Camarilla levels (calculated from previous 12h bar)
    # We'll calculate these inside the loop using prior bar data
    # But to avoid look-ahead, we use the previous completed 12h bar's OHLC
    
    # Volume spike: current volume > 1.5 * average volume of last 20 periods
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need 1d EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from previous 12h bar
        # Need to find index of previous completed 12h bar
        # Since we're on 12h timeframe, previous bar is i-1
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            if range_ > 0:  # avoid division by zero
                # Camarilla levels
                R3 = prev_close + (range_ * 1.1 / 4)
                S3 = prev_close - (range_ * 1.1 / 4)
                
                # Volume confirmation
                vol_ma = volume_ma[i]
                vol_spike = volume[i] > 1.5 * vol_ma if not np.isnan(vol_ma) else False
                
                if position == 0:
                    # Long: price breaks above R3 with volume spike + 1d uptrend
                    if close[i] > R3 and vol_spike and close[i] > ema34_1d_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: price breaks below S3 with volume spike + 1d downtrend
                    elif close[i] < S3 and vol_spike and close[i] < ema34_1d_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                elif position == 1:
                    # Exit long when price closes below previous day's close (mean reversion)
                    if close[i] < prev_close:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit short when price closes above previous day's close
                    if close[i] > prev_close:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # No range, hold or flat
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data for previous bar
            signals[i] = 0.0
    
    return signals