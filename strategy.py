#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# - Long when price breaks above 4h Donchian(20) high in 1d uptrend (close > EMA200) with volume spike
# - Short when price breaks below 4h Donchian(20) low in 1d downtrend (close < EMA200) with volume spike
# - Exit: ATR trailing stop (highest high since entry - 2.5*ATR for longs, lowest low + 2.5*ATR for shorts)
# - Uses discrete position sizing (0.30) to minimize fee churn
# - Targets 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# - Works in bull (breakouts) and bear (short breakdowns) via symmetric logic

name = "4h_1d_donchian_breakout_volume_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average (stricter to reduce trades)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_4h = np.zeros_like(tr)
    atr_14_4h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_4h[i] = (atr_14_4h[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    for i in range(100, n):
        close_4h = prices['close'].iloc[i]
        
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
            # Exit: ATR trailing stop
            if close_4h < highest_since_entry - 2.5 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
            # Exit: ATR trailing stop
            if close_4h > lowest_since_entry + 2.5 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: break above Donchian high in 1d uptrend
                if (close_4h > donchian_high[i] and 
                    close_4h > ema_200_1d_aligned[i]):
                    position = 1
                    entry_price = close_4h
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.30
                # Short signal: break below Donchian low in 1d downtrend
                elif (close_4h < donchian_low[i] and 
                      close_4h < ema_200_1d_aligned[i]):
                    position = -1
                    entry_price = close_4h
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.30
    
    return signals