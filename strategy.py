#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Calculate Camarilla levels from previous day
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    range_hl = prev_high - prev_low
    R1 = prev_close + range_hl * 1.1 / 12
    S1 = prev_close - range_hl * 1.1 / 12
    
    # 1-day trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    daily_up = close > ema34_1d_aligned
    daily_down = close < ema34_1d_aligned
    
    # Volume filter: volume > 1.5x 20-period SMA
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_sma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~8 hours to reduce trade frequency
    
    start_idx = max(20, 34)  # Ensure enough data for volume SMA and daily EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R1, daily uptrend, volume spike
            if close[i] > R1[i] and daily_up[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S1, daily downtrend, volume spike
            elif close[i] < S1[i] and daily_down[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price breaks below S1 OR daily trend turns down
            if close[i] < S1[i] or not daily_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above R1 OR daily trend turns up
            if close[i] > R1[i] or not daily_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance. 
# Breakouts with volume confirmation and daily trend filter capture momentum moves.
# Long when price breaks above R1 in daily uptrend with volume spike.
# Short when price breaks below S1 in daily downtrend with volume spike.
# Exit when price reverses to opposite level or trend changes.
# This strategy avoids counter-trend trades and uses volume to confirm breakout strength.
# Position size 0.25 manages risk, cooldown of 2 bars (~8h) limits trades to ~20-40/year.
# Works in bull markets (captures uptrend continuations) and bear markets (captures downtrend continuations).