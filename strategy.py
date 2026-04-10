#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume spike and trend filter
# - Long when price breaks above 20-period Donchian high on 4h in 1d uptrend (close > EMA50) with 1d volume spike
# - Short when price breaks below 20-period Donchian low on 4h in 1d downtrend (close < EMA50) with 1d volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss to limit drawdown
# - Targets 20-50 trades/year (75-200 total over 4 years) to avoid fee drag
# - Daily trend filter reduces false breakouts in ranging markets

name = "4h_1d_donchian_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute ATR for stoploss (using 1d data)
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1d = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate Donchian channels on 4h data (20-period)
            if i >= 20:  # Need enough data for Donchian
                donchian_high = np.max(prices['high'].iloc[i-20:i].values)
                donchian_low = np.min(prices['low'].iloc[i-20:i].values)
                
                # Long signal: price breaks above Donchian high in 1d uptrend with volume spike
                if (prices['high'].iloc[i] > donchian_high and 
                    prices['close'].iloc[i] > ema_50_1d_aligned[i] and 
                    vol_spike_1d_aligned[i]):
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low in 1d downtrend with volume spike
                elif (prices['low'].iloc[i] < donchian_low and 
                      prices['close'].iloc[i] < ema_50_1d_aligned[i] and 
                      vol_spike_1d_aligned[i]):
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals