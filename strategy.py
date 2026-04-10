#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR regime filter
# - Primary: 4h timeframe for balance of trade frequency and signal quality
# - Entry: Price breaks above/below 20-period Donchian channel + volume > 1.5x 20-period MA + ATR(14) > 30th percentile
# - Exit: Opposite Donchian channel break (mean reversion) or ATR trailing stop (3x ATR)
# - Position sizing: 0.25 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) - within 4h sweet spot
# - Works in bull/bear: Donchian breakouts capture trends, volume/ATR filters avoid false signals in chop

name = "4h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian Channel (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for volatility regime filter
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h) - pd.Series(close_4h).shift(1))
    tr3 = abs(pd.Series(low_4h) - pd.Series(close_4h).shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h ATR percentile rank (using 30-period lookback)
    atr_percentile = pd.Series(atr_4h).rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 4h volume moving average (20-period) for volume confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_percentile[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Volatility regime: ATR > 30th percentile (avoid extremely low volatility)
        vol_regime = atr_percentile[i] > 30
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_4h[i] > 1.5 * volume_ma_20_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above upper Donchian + vol regime + volume spike
            if (close_4h[i] > highest_20[i] and vol_regime and volume_spike):
                position = 1
                entry_price = close_4h[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian + vol regime + volume spike
            elif (close_4h[i] < lowest_20[i] and vol_regime and volume_spike):
                position = -1
                entry_price = close_4h[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:  # Long position
                highest_since_entry = max(highest_since_entry, high_4h[i])
                lowest_since_entry = min(lowest_since_entry, low_4h[i])
                
                # Exit conditions:
                # 1. Price breaks below lower Donchian (mean reversion)
                # 2. ATR trailing stop: price drops 3*ATR from highest since entry
                exit_condition = (
                    close_4h[i] < lowest_20[i] or
                    close_4h[i] < highest_since_entry - 3.0 * atr_4h[i]
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, high_4h[i])
                lowest_since_entry = min(lowest_since_entry, low_4h[i])
                
                # Exit conditions:
                # 1. Price breaks above upper Donchian (mean reversion)
                # 2. ATR trailing stop: price rises 3*ATR from lowest since entry
                exit_condition = (
                    close_4h[i] > highest_20[i] or
                    close_4h[i] > lowest_since_entry + 3.0 * atr_4h[i]
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals