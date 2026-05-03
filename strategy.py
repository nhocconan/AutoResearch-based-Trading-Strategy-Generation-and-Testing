#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses 12h timeframe for lower trade frequency (target: 50-150 total trades over 4 years).
# Donchian channels from prior completed 12h bar provide structural breakout levels.
# 1d EMA50 determines trend direction to avoid counter-trend trades.
# Volume spike confirms institutional participation. Discrete sizing 0.25 to manage drawdown.

name = "12h_Donchian20_1dEMA50_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation and 1d data for trend/volume filters
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior completed 12h bar's OHLC for Donchian calculation
    prior_high_12h = np.roll(df_12h['high'].values, 1)
    prior_low_12h = np.roll(df_12h['low'].values, 1)
    prior_high_12h[0] = np.nan
    prior_low_12h[0] = np.nan
    
    # Calculate Donchian channels for prior 12h bar (20-period)
    upper_chan = pd.Series(prior_high_12h).rolling(window=20, min_periods=20).max().values
    lower_chan = pd.Series(prior_low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_chan_aligned = align_htf_to_ltf(prices, df_12h, upper_chan)
    lower_chan_aligned = align_htf_to_ltf(prices, df_12h, lower_chan)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume regime (high volume when current volume > 1.5x 20-period MA)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = vol_1d > (1.5 * vol_ma_1d)  # High volume regime
    
    # Align volume regime to 12h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate ATR(14) for 12h data (for stoploss)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(50, n):
        # Get current values
        upper = upper_chan_aligned[i]
        lower = lower_chan_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_reg = vol_regime_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(upper) or np.isnan(lower) or np.isnan(ema_trend) or np.isnan(vol_reg) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        # Entry conditions
        # Long: break above upper channel with volume spike, above 1d EMA50, and in high volume regime
        long_entry = (close[i] > upper) and volume_spike and (close[i] > ema_trend) and vol_reg
        # Short: break below lower channel with volume spike, below 1d EMA50, and in high volume regime
        short_entry = (close[i] < lower) and volume_spike and (close[i] < ema_trend) and vol_reg
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals