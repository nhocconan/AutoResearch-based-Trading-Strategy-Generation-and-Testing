#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d ATR ratio < 0.8 (low volatility) AND volume > 1.2x average
# - Short when price breaks below 20-period Donchian low AND 1d ATR ratio < 0.8 AND volume > 1.2x average
# - Exit when price crosses the 10-period Donchian midpoint (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian breakouts capture momentum bursts in ranging markets
# - Low volatility filter (ATR ratio) avoids false breakouts during high volatility
# - Volume confirmation ensures breakouts have conviction

name = "4h_1d_donchian_volatility_filter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) for entry
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    
    # Donchian(10) for exit (midpoint)
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid = (highest_high_10 + lowest_low_10) / 2
    
    # Pre-compute 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * vol_ma)
    
    # Pre-compute 1d ATR-based volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period ATR average
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)  # fill NaN with 1.0
    
    # Low volatility filter: ATR ratio < 0.8
    low_volatility = atr_ratio < 0.8
    
    # Align HTF indicators to 4h timeframe
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(low_volatility_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND low volatility AND volume spike
            if (close[i] > donchian_high[i] and 
                low_volatility_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND low volatility AND volume spike
            elif (close[i] < donchian_low[i] and 
                  low_volatility_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses the 10-period Donchian midpoint (mean reversion)
            exit_long = (position == 1 and close[i] < donchian_mid[i])
            exit_short = (position == -1 and close[i] > donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals