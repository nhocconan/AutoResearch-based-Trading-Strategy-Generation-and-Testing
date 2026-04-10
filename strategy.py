#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# - Long when price breaks above 4h Donchian(20) upper band in 12h uptrend (close > EMA50) with volume spike
# - Short when price breaks below 4h Donchian(20) lower band in 12h downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or price reverts to Donchian midpoint
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in both bull (trend continuation) and bear (mean reversion to midpoint) regimes

name = "4h_12h_donchian_breakout_volume_trend_atr_v1"
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
    
    # 4h Donchian(20) channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # 4h ATR(14) for stoploss
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
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > donchian_middle[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < donchian_middle[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike_12h_aligned[i]:
                # Long signal: price breaks above upper band in 12h uptrend
                if (prices['high'].iloc[i] > donchian_upper[i] and 
                    prices['close'].iloc[i] > ema_50_12h_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below lower band in 12h downtrend
                elif (prices['low'].iloc[i] < donchian_lower[i] and 
                      prices['close'].iloc[i] < ema_50_12h_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h[i]
                    signals[i] = -0.25
    
    return signals