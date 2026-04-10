#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# - Long when price breaks above Donchian(20) high with 1d uptrend (close > EMA50) and volume spike (>1.5x 20-bar avg)
# - Short when price breaks below Donchian(20) low with 1d downtrend (close < EMA50) and volume spike
# - ATR(14) stoploss: exit when price moves against position by 2.5x ATR
# - Discrete position sizing 0.25 to minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to avoid overtrading and fee drag

name = "4h_1d_donchian_volume_trend_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 4h Donchian channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20) upper and lower bands
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = np.zeros_like(tr)
    atr_14[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below Donchian low
            if (prices['close'].iloc[i] < entry_price - 2.5 * entry_atr or 
                prices['close'].iloc[i] < donch_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above Donchian high
            if (prices['close'].iloc[i] > entry_price + 2.5 * entry_atr or 
                prices['close'].iloc[i] > donch_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above Donchian high in 1d uptrend
                if prices['close'].iloc[i] > donch_high[i] and prices['close'].iloc[i] > ema_50_1d_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low in 1d downtrend
                elif prices['close'].iloc[i] < donch_low[i] and prices['close'].iloc[i] < ema_50_1d_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14[i]
                    signals[i] = -0.25
    
    return signals