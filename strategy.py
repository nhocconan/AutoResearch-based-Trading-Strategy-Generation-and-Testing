#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions (<-80 oversold, >-20 overbought)
# 1d EMA34 ensures alignment with daily trend for higher probability mean-reversion trades
# Volume spike (2x 20-period average) confirms institutional participation during reversals
# Works in bull markets via buying oversold dips and in bear markets via selling overbought rallies
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period) on prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need at least 15 days for Williams %R calculation
        return np.zeros(n)
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Use prior completed 1d bar's Williams %R to avoid look-ahead
    williams_r_prior = np.roll(williams_r, 1)
    williams_r_prior[0] = np.nan  # First value has no prior
    
    # Williams %R extremes: <-80 oversold, >-20 overbought
    williams_oversold = williams_r_prior < -80
    williams_overbought = williams_r_prior > -20
    
    # Calculate 1d EMA34 for trend filter (prior completed 1d bar)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prior = np.roll(ema_34_1d, 1)
    ema_34_1d_prior[0] = np.nan
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 6h timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold)
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_prior)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold AND price > 1d EMA34 (bullish bias) AND volume spike
            if (williams_oversold_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought AND price < 1d EMA34 (bearish bias) AND volume spike
            elif (williams_overbought_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R becomes overbought OR price falls below 1d EMA34 (trend change)
            if williams_overbought_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R becomes oversold OR price rises above 1d EMA34 (trend change)
            if williams_oversold_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals