#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold) in 1d uptrend with volume spike (>2.0x 20-period volume MA).
# Short when Williams %R crosses below -20 (overbought) in 1d downtrend with volume spike.
# Williams %R identifies exhaustion points in ranging markets. 1d EMA34 ensures higher timeframe alignment.
# Volume spike confirms institutional participation. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.

name = "6h_WilliamsR_1dEMA34_VolumeSpike"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    
    # Align Williams %R to lower timeframe (6h is primary timeframe, so no alignment needed)
    # But we need to align the 6h values to the 6h bars themselves (identity)
    williams_r_aligned = williams_r  # Already at 6h frequency
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        williams_r_val = williams_r_aligned[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND 1d uptrend AND volume spike
            if williams_r_val > -80 and williams_r_val < -20 and trend_up and vol_spike:
                # Check for crossover: previous value was <= -80
                if i > 100 and williams_r_aligned[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_val
            # Short: Williams %R crosses below -20 (from above) AND 1d downtrend AND volume spike
            elif williams_r_val < -20 and williams_r_val > -80 and trend_down and vol_spike:
                # Check for crossover: previous value was >= -20
                if i > 100 and williams_r_aligned[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: Williams %R crosses below -50 (momentum loss)
            if williams_r_val < -50 and williams_r_aligned[i-1] >= -50:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: Williams %R crosses above -50 (momentum loss)
            if williams_r_val > -50 and williams_r_aligned[i-1] <= -50:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals