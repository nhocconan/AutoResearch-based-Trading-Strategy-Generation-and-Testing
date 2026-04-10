#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with 1d uptrend (close > EMA20) and volume spike
# - Short when price breaks below Camarilla L3 level with 1d downtrend (close < EMA20) and volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Daily trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown

name = "4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
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
    
    # 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute ATR for stoploss (using 4h data)
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14 = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate Camarilla pivot levels on 1d data (using previous completed 1d bar)
            # Each 4h bar = 1/6 of a day, so 6 bars per day
            completed_1d_bars = i // 6  # bars per day
            if completed_1d_bars >= 1 and completed_1d_bars < len(df_1d):
                # Use previous completed 1d bar for Camarilla calculation
                prev_1d_idx = completed_1d_bars - 1
                if prev_1d_idx >= 0:
                    high = df_1d['high'].iloc[prev_1d_idx]
                    low = df_1d['low'].iloc[prev_1d_idx]
                    close = df_1d['close'].iloc[prev_1d_idx]
                    
                    # Calculate Camarilla levels
                    range_val = high - low
                    camarilla_h3 = close + range_val * 1.1 / 4
                    camarilla_l3 = close - range_val * 1.1 / 4
                    
                    # Long signal: price breaks above Camarilla H3 in 1d uptrend with volume spike
                    if (prices['high'].iloc[i] > camarilla_h3 and 
                        prices['close'].iloc[i] > ema_20_1d_aligned[i] and 
                        vol_spike_1d_aligned[i]):
                        position = 1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = 0.25
                    # Short signal: price breaks below Camarilla L3 in 1d downtrend with volume spike
                    elif (prices['low'].iloc[i] < camarilla_l3 and 
                          prices['close'].iloc[i] < ema_20_1d_aligned[i] and 
                          vol_spike_1d_aligned[i]):
                        position = -1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = -0.25
    
    return signals