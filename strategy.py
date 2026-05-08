#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123 Reversal pattern with 1d trend filter and volume confirmation
# Long when: 4h low < previous low AND close > previous close (123 buy)
#            AND 1d EMA50 rising AND volume > 1.3x average
# Short when: 4h high > previous high AND close < previous close (123 sell)
#             AND 1d EMA50 falling AND volume > 1.3x average
# Exit on opposite 123 signal or trend change
# Targets 50-150 total trades over 4 years (12-38/year) for low fee drag

name = "4h_123Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for 123 pattern (already 4h timeframe)
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # need at least 2 bars for 123 pattern
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 123 pattern conditions
        # Buy setup: current low < previous low AND current close > previous close
        buy_setup = (low[i] < low[i-1]) and (close[i] > close[i-1])
        # Sell setup: current high > previous high AND current close < previous close
        sell_setup = (high[i] > high[i-1]) and (close[i] < close[i-1])
        
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: 123 buy setup, 1d uptrend, volume spike
            if buy_setup and ema50_1d_val > ema50_1d_aligned[i-1] and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: 123 sell setup, 1d downtrend, volume spike
            elif sell_setup and ema50_1d_val < ema50_1d_aligned[i-1] and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 123 sell setup or 1d trend down
            if sell_setup or ema50_1d_val < ema50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 123 buy setup or 1d trend up
            if buy_setup or ema50_1d_val > ema50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals