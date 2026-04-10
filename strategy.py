#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high in 12h uptrend (close > EMA50) with volume spike (>1.5x 20-bar avg)
# - Short when price breaks below Donchian(20) low in 12h downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - 12h trend filter reduces false breakouts in ranging markets
# - ATR-based stoploss to limit drawdown

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume confirmation: > 1.5x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute ATR for stoploss (using 1h data for reasonable granularity)
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
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
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
            # Calculate Donchian(20) levels on 4h data (using previous completed bar)
            if i >= 20:
                # Use data up to previous bar for Donchian calculation (no look-ahead)
                lookback_start = i - 20
                lookback_end = i  # exclusive, so we get bars [i-20, i-1]
                if lookback_start >= 0:
                    high_20 = prices['high'].iloc[lookback_start:lookback_end].max()
                    low_20 = prices['low'].iloc[lookback_start:lookback_end].min()
                    
                    # Long signal: price breaks above Donchian high in 12h uptrend with volume spike
                    if (prices['close'].iloc[i] > high_20 and 
                        prices['close'].iloc[i] > ema_50_12h_aligned[i] and 
                        vol_spike_12h_aligned[i]):
                        position = 1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = 0.25
                    # Short signal: price breaks below Donchian low in 12h downtrend with volume spike
                    elif (prices['close'].iloc[i] < low_20 and 
                          prices['close'].iloc[i] < ema_50_12h_aligned[i] and 
                          vol_spike_12h_aligned[i]):
                        position = -1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = -0.25
    
    return signals