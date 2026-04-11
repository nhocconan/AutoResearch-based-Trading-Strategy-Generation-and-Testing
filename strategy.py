#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_extreme_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate CCI on daily timeframe (20-period)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std()
    cci = (typical_price - tp_mean) / (0.015 * tp_std)
    cci = cci.values
    
    # Align daily CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Volume confirmation: volume > 2x 20-period average (more stringent)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(cci_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        cci_val = cci_aligned[i]
        
        # Volume confirmation (more stringent: 2x average)
        volume_confirmed = volume_current > 2.0 * vol_ma_20[i]
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: CCI < -100 (oversold) + volume confirmation
        if cci_val < -100 and volume_confirmed:
            long_signal = True
        
        # Short: CCI > 100 (overbought) + volume confirmation
        if cci_val > 100 and volume_confirmed:
            short_signal = True
        
        # Exit conditions: CCI returns to neutral zone (-50 to 50)
        exit_long = position == 1 and cci_val > -50
        exit_short = position == -1 and cci_val < 50
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
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

# Hypothesis: CCI extreme + volume strategy on 6h timeframe.
# Uses daily CCI(20) to identify extreme overbought/oversold conditions (>100/< -100).
# Requires volume confirmation (>2x 20-period average) to ensure institutional participation.
# Enters long on extreme oversold with volume, short on extreme overbought with volume.
# Exits when CCI returns to neutral zone (-50 to 50).
# Works in both bull and bear markets by trading mean reversion from extremes.
# Designed for 6h timeframe with selective entries to target 50-150 total trades over 4 years.