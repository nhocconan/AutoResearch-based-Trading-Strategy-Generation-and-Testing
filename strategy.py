# [76720] 4H_Range_Breakout_1dTrend_Volume
# Hypothesis: 4-hour Candlestick Range Breakout with 1-day Trend Filter and Volume Confirmation.
# Long when price breaks above the prior 4-hour bar's high during 1-day uptrend with volume spike.
# Short when price breaks below the prior 4-hour bar's low during 1-day downtrend with volume spike.
# Exit when price returns to the prior 4-hour bar's midpoint or trend reverses.
# Designed for moderate trade frequency by requiring trend alignment and volume confirmation.
# Works in both bull and bear markets by following the 1-day trend.

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
    
    # Prior 4-hour bar's high and low for breakout detection
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_mid = (prior_high + prior_low) / 2
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_mid[0] = np.nan
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above prior 4h high + 1d uptrend + volume spike
            if close[i] > prior_high[i] and ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 4h low + 1d downtrend + volume spike
            elif close[i] < prior_low[i] and ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to prior 4h midpoint or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below midpoint or 1d trend turns down
                if close[i] < prior_mid[i] or ema20_1d_aligned[i] < ema20_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above midpoint or 1d trend turns up
                if close[i] > prior_mid[i] or ema20_1d_aligned[i] > ema20_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Range_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0