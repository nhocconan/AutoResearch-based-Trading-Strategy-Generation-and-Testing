#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high in 1w uptrend (close > EMA50) with volume spike
# - Short when price breaks below Donchian(20) low in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or price reverts to Donchian midpoint
# - Targets 15-30 trades/year (60-120 total over 4 years) to avoid fee drag

name = "1d_1w_donchian_breakout_volume_trend_atr_v1"
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
    
    # 1w ATR(14) for stoploss
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = np.zeros_like(tr)
    atr_14_1w[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1w[i] = (atr_14_1w[i-1] * (14-1) + tr[i]) / 14
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # 1w volume confirmation: > 1.5x 10-period average
    avg_volume_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_spike_1w = volume_1w > (1.5 * avg_volume_10_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1d Donchian(20) channels
    high_roll_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_roll_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll_20 + low_roll_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(high_roll_20[i]) or np.isnan(low_roll_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_1w_aligned[i]:
                # Long signal: price breaks above Donchian high in 1w uptrend
                if (prices['high'].iloc[i] > high_roll_20[i] and 
                    prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1w_aligned[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low in 1w downtrend
                elif (prices['low'].iloc[i] < low_roll_20[i] and 
                      prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1w_aligned[i]
                    signals[i] = -0.25
    
    return signals