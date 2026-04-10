#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level from prior 1d in 1w uptrend (close > EMA50) with volume > 1.8x 24-bar avg
# - Short when price breaks below Camarilla L3 level from prior 1d in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - 1w trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown

name = "12h_1w_camarilla_pivot_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Camarilla pivot levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    camarilla_h3 = pivot + (range_1d * 1.1 / 4)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 12h volume confirmation: > 1.8x 24-period average
    volume_12h = get_htf_data(prices, '12h')['volume'].values
    avg_volume_24_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    vol_spike_12h = volume_12h > (1.8 * avg_volume_24_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), vol_spike_12h)
    
    # Pre-compute ATR for stoploss (using 12h data)
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
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
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
            # Long signal: price breaks above Camarilla H3 in 1w uptrend with volume spike
            if (prices['close'].iloc[i] > camarilla_h3_aligned[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short signal: price breaks below Camarilla L3 in 1w downtrend with volume spike
            elif (prices['close'].iloc[i] < camarilla_l3_aligned[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals