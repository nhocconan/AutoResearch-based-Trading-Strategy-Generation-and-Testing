#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d volume spike + 1d ATR regime filter
# - Primary signal: Price breaks above/below 20-period Donchian channel on 6h
# - Volume confirmation: 1d volume > 1.8x 20-period average volume (strict filter for low-fee trades)
# - Regime filter: 1d ATR(14) > 1.5x ATR(50) (high volatility regime) to avoid choppy markets
# - Works in bull/bear: In high volatility, breakouts have follow-through; avoids low-vol fakeouts
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "6h_1d_donchian_volume_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1d ATR regime filter: ATR(14) > 1.5 * ATR(50) = high volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    high_vol_regime = atr_14 > (1.5 * atr_50)
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Pre-compute 6h Donchian Channel (20)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 6h ATR(20) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(high_vol_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] < donchian_mid[i] or close_6h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] > donchian_mid[i] or close_6h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume spike and high volatility regime
            if volume_spike_aligned[i] and high_vol_aligned[i]:
                # Long: price breaks above upper Donchian band
                if close_6h[i] > donchian_high[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band
                elif close_6h[i] < donchian_low[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals