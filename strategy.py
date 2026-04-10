#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level with 12h uptrend (close > EMA30) and volume spike
# - Short when price breaks below Camarilla L3 level with 12h downtrend (close < EMA30) and volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - 12h trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown

name = "4h_12h_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(30) for trend filter
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # 12h volume confirmation: > 1.8x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.8 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute ATR for stoploss (using 1h data for finer granularity)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    high_low = df_1h['high'] - df_1h['low']
    high_close = np.abs(df_1h['high'] - df_1h['close'].shift(1))
    low_close = np.abs(df_1h['low'] - df_1h['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1h = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    atr_14_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_14_1h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_30_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(atr_14_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14_1h_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14_1h_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate Camarilla pivot levels on 1d data (using previous completed 1d bar)
            # 4h bars per day = 6
            completed_1d_bars = i // 6
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
                    
                    # Long signal: price breaks above Camarilla H3 in 12h uptrend with volume spike
                    if (prices['high'].iloc[i] > camarilla_h3 and 
                        prices['close'].iloc[i] > ema_30_12h_aligned[i] and 
                        vol_spike_12h_aligned[i]):
                        position = 1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = 0.25
                    # Short signal: price breaks below Camarilla L3 in 12h downtrend with volume spike
                    elif (prices['low'].iloc[i] < camarilla_l3 and 
                          prices['close'].iloc[i] < ema_30_12h_aligned[i] and 
                          vol_spike_12h_aligned[i]):
                        position = -1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = -0.25
    
    return signals