#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily volatility-adjusted price channels with volume confirmation
# - Long when price breaks above upper channel (mean + 2*ATR) with volume > 1.5x 20-period average
# - Short when price breaks below lower channel (mean - 2*ATR) with volume > 1.5x 20-period average
# - Uses 1d ATR for volatility normalization to adapt to changing market conditions
# - Volume confirmation ensures institutional participation at breakouts
# - Target: 50-150 total trades over 4 years (12-37/year) for optimal frequency
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate daily ATR (14-period)
    tr_d = np.maximum(
        high_d[1:] - low_d[1:],
        np.maximum(
            np.abs(high_d[1:] - close_d[:-1]),
            np.abs(low_d[1:] - close_d[:-1])
        )
    )
    tr_d = np.concatenate([[np.nan], tr_d])  # Align with original index
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily mean price (typical price)
    typical_price_d = (high_d + low_d + close_d) / 3
    
    # Calculate upper and lower channels (mean ± 2*ATR)
    upper_channel_d = typical_price_d + 2 * atr_d
    lower_channel_d = typical_price_d - 2 * atr_d
    
    # Align daily channels to 12h timeframe
    upper_channel_12h = align_htf_to_ltf(prices, df_d, upper_channel_d)
    lower_channel_12h = align_htf_to_ltf(prices, df_d, lower_channel_d)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]) or np.isnan(upper_channel_12h[i]) or np.isnan(lower_channel_12h[i]):
            continue
        
        if position == 0:
            # Long: Price breaks above upper channel with volume confirmation
            if close[i] > upper_channel_12h[i] and volume[i] > vol_ma[i] * 1.5:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower channel with volume confirmation
            elif close[i] < lower_channel_12h[i] and volume[i] > vol_ma[i] * 1.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price returns to mean or opposite channel
            typical_price = (high[i] + low[i] + close[i]) / 3
            if typical_price < upper_channel_12h[i] or close[i] < lower_channel_12h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price returns to mean or opposite channel
            typical_price = (high[i] + low[i] + close[i]) / 3
            if typical_price > lower_channel_12h[i] or close[i] > upper_channel_12h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_VolatilityChannel_VolumeBreakout"
timeframe = "12h"
leverage = 1.0