#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA50) and volume confirmation
# - Long when price breaks above 20-period Donchian high on 1d in 1w uptrend (close > EMA50_1w) with volume spike (>1.5x 20-day avg volume)
# - Short when price breaks below 20-period Donchian low on 1d in 1w downtrend (close < EMA50_1w) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR(14) stoploss at 2.5x ATR to limit drawdown
# - Targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag
# - Weekly trend filter reduces false breakouts in ranging markets, works in both bull and bear

name = "1d_1w_donchian_breakout_volume_trend_v1"
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume confirmation: > 1.5x 20-period average
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (1.5 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Pre-compute ATR for stoploss
    high_low = df_1w['high'] - df_1w['low']
    high_close = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    low_close = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1w = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14_1w_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14_1w_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate Donchian channels on 1w data (20-period)
            if i >= 20:  # Need enough data for Donchian
                # Get the index of the completed 1w bar
                completed_1w_bars = i // (7*24*4)  # Assuming 15m bars, 7*24*4=672 bars per 1w
                if completed_1w_bars < len(df_1w) and completed_1w_bars >= 20:
                    # Calculate Donchian for completed 1w bars
                    start_idx = max(0, completed_1w_bars - 20)
                    end_idx = completed_1w_bars
                    donchian_high_val = np.max(high_1w[start_idx:end_idx])
                    donchian_low_val = np.min(low_1w[start_idx:end_idx])
                    
                    # Long signal: price breaks above Donchian high in 1w uptrend with volume spike
                    if (prices['high'].iloc[i] > donchian_high_val and 
                        prices['close'].iloc[i] > ema_50_1w_aligned[i] and 
                        vol_spike_1w_aligned[i]):
                        position = 1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = 0.25
                    # Short signal: price breaks below Donchian low in 1w downtrend with volume spike
                    elif (prices['low'].iloc[i] < donchian_low_val and 
                          prices['close'].iloc[i] < ema_50_1w_aligned[i] and 
                          vol_spike_1w_aligned[i]):
                        position = -1
                        entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                        signals[i] = -0.25
    
    return signals