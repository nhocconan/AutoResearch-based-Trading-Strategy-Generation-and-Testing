#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level in 1w uptrend (close > EMA50) with volume spike
# - Short when price breaks below Camarilla L3 level in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown
# - Designed to work in both bull and bear markets via trend filter

name = "1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume confirmation: > 2.0x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Pre-compute ATR for stoploss (using 1d data)
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
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
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
            # Need at least one completed 1d bar
            completed_1d_bars = i // 24  # 24 bars per day (1h bars in 1d timeframe?)
            # Actually, for 1d timeframe, each bar is 1 day, so:
            completed_1d_bars = i  # each index is a completed day
            if i >= 1:  # Need at least one completed 1d bar (yesterday)
                # Use previous completed 1d bar for Camarilla calculation
                prev_1d_idx = i - 1
                if prev_1d_idx >= 0 and prev_1d_idx < len(prices):
                    high = prices['high'].iloc[prev_1d_idx]
                    low = prices['low'].iloc[prev_1d_idx]
                    close = prices['close'].iloc[prev_1d_idx]
                    
                    # Calculate Camarilla levels
                    range_val = high - low
                    camarilla_h3 = close + range_val * 1.1 / 4
                    camarilla_l3 = close - range_val * 1.1 / 4
                    
                    # Long signal: price breaks above Camarilla H3 in 1w uptrend with volume spike
                    if (prices['high'].iloc[i] > camarilla_h3 and 
                        prices['close'].iloc[i] > ema_50_1w_aligned[i] and 
                        vol_spike_1w_aligned[i]):
                        position = 1
                        entry_price = prices['open'].iloc[i]  # enter at open of signal bar
                        signals[i] = 0.25
                    # Short signal: price breaks below Camarilla L3 in 1w downtrend with volume spike
                    elif (prices['low'].iloc[i] < camarilla_l3 and 
                          prices['close'].iloc[i] < ema_50_1w_aligned[i] and 
                          vol_spike_1w_aligned[i]):
                        position = -1
                        entry_price = prices['open'].iloc[i]  # enter at open of signal bar
                        signals[i] = -0.25
    
    return signals