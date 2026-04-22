#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(21) trend filter and volume confirmation.
# Uses weekly EMA for trend direction, daily Donchian breakout for entry, and volume spike for confirmation.
# Long when price breaks above daily Donchian upper in uptrend (close > weekly EMA21) with volume spike.
# Short when price breaks below daily Donchian lower in downtrend (close < weekly EMA21) with volume spike.
# Exit on opposite Donchian band touch or trend reversal.
# Designed for 1d timeframe to target 15-30 trades/year per symbol.
# Works in bull/bear via trend filter + volatility-based entry levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(21) for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to 1d timeframe (waits for 1w bar to close)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + uptrend (close > weekly EMA21) + volume spike
            if (close[i] > high_roll[i] and 
                close[i] > ema_21_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + downtrend (close < weekly EMA21) + volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < ema_21_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian lower touch or trend reversal
                if (close[i] < low_roll[i] or close[i] < ema_21_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian upper touch or trend reversal
                if (close[i] > high_roll[i] or close[i] > ema_21_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA21_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0