#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed weekly bars
    upper_1w = np.roll(upper_1w, 1)
    lower_1w = np.roll(lower_1w, 1)
    upper_1w[0] = np.nan
    lower_1w[0] = np.nan
    
    # Align weekly indicators to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Calculate daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.01 * price_close  # ATR > 1% of price
        
        # Long conditions: price breaks above weekly upper Donchian with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_high > upper_1w_aligned[i])
        
        # Short conditions: price breaks below weekly lower Donchian with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_low < lower_1w_aligned[i])
        
        # Exit when price returns to the midpoint of the weekly channel
        mid_1w = (upper_1w_aligned[i] + lower_1w_aligned[i]) / 2
        exit_long = position == 1 and price_close < mid_1w
        exit_short = position == -1 and price_close > mid_1w
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Donchian breakout with volume and volatility filters on daily timeframe.
# Uses 20-week Donchian channels to identify major support/resistance levels.
# Enters long when daily price breaks above weekly upper channel with volume confirmation (>1.5x average)
# and sufficient volatility (ATR > 1% of price). Enters short when price breaks below weekly lower channel
# under same conditions. Exits when price returns to the midpoint of the weekly channel.
# Works in both bull and bear markets by capturing major breakouts. Target: 20-80 total trades
# over 4 years (5-20/year) to minimize fee drag on daily timeframe. Weekly timeframe reduces noise
# and captures multi-week trends. Volume confirmation ensures institutional participation.
# Volatility filter prevents whipsaws in low-volatility environments. Midpoint exit provides
# systematic profit-taking while allowing trends to develop.