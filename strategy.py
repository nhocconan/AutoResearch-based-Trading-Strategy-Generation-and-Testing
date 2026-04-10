#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike filter and 1w ATR-based volatility regime
# - Primary signal: Price breaks above/below 20-period Donchian channel on 4h
# - Volume confirmation: 1d volume > 1.8x 20-period average volume (strict filter to reduce trades)
# - Regime filter: 1w ATR(14) / SMA(close, 50) > 0.02 indicates high volatility (favor breakouts)
# - Works in bull/bear: High volatility regime captures breakouts in both directions
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "4h_1d_1w_donchian_volume_volatility_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter (strict: 1.8x average)
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1w volatility regime: ATR(14) / SMA(close, 50)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # SMA(close, 50)
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Volatility ratio: ATR/SMA (avoid division by zero)
    vol_ratio = np.where(sma_50 > 0, atr_14 / sma_50, 0)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio, additional_delay_bars=0)
    
    # Pre-compute 4h Donchian Channel (20)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] < donchian_mid[i] or close_4h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] > donchian_mid[i] or close_4h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume spike and high volatility regime
            # Only trade in high volatility regimes (vol_ratio > 0.02) to avoid choppy markets
            if volume_spike_aligned[i] and vol_ratio_aligned[i] > 0.02:
                # Long: price breaks above upper Donchian band
                if close_4h[i] > donchian_high[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian band
                elif close_4h[i] < donchian_low[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals