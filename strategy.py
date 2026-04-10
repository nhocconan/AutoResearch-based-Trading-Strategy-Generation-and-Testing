#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level in 1d uptrend (close > EMA50) with volume > 1.8x 20-bar avg
# - Short when price breaks below Camarilla L3 level in 1d downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - 1d trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown

name = "4h_1d_camarilla_pivot_breakout_volume_trend_v1"
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
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
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
            # Calculate Camarilla pivot levels from previous 1d bar
            if i >= 24:  # Need at least one completed 1d bar (24*4h bars per day)
                # Previous completed 1d bar index in 4h data
                prev_1d_idx = ((i // 24) * 24) - 1
                if prev_1d_idx >= 0:
                    # Get previous day's OHLC from 4h data (last 24 bars of previous day)
                    day_start = prev_1d_idx - 23
                    if day_start >= 0:
                        day_high = prices['high'].iloc[day_start:prev_1d_idx+1].max()
                        day_low = prices['low'].iloc[day_start:prev_1d_idx+1].min()
                        day_close = prices['close'].iloc[prev_1d_idx]
                        
                        # Calculate Camarilla levels
                        range_ = day_high - day_low
                        if range_ > 0:
                            camarilla_h3 = day_close + (range_ * 1.1 / 4)
                            camarilla_l3 = day_close - (range_ * 1.1 / 4)
                            
                            # Long signal: price breaks above Camarilla H3 in 1d uptrend with volume spike
                            if (prices['close'].iloc[i] > camarilla_h3 and 
                                prices['close'].iloc[i] > ema_50_1d_aligned[i] and 
                                vol_spike_1d_aligned[i]):
                                position = 1
                                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                                signals[i] = 0.25
                            # Short signal: price breaks below Camarilla L3 in 1d downtrend with volume spike
                            elif (prices['close'].iloc[i] < camarilla_l3 and 
                                  prices['close'].iloc[i] < ema_50_1d_aligned[i] and 
                                  vol_spike_1d_aligned[i]):
                                position = -1
                                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                                signals[i] = -0.25
    
    return signals