#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR filter
# - Primary signal: Price breaks above/below Donchian(20) channel on daily
# - Volume filter: 1w volume > 1.5x 10-period average volume (institutional participation)
# - ATR filter: 1w ATR(14) < 0.03 * price (low volatility for cleaner breakouts)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(20) on 1d
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Breakouts capture strong moves; filters avoid chop/false signals

name = "1d_1w_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w volume spike filter
    volume_1w = df_1w['volume'].values
    avg_volume_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_spike = volume_1w > (1.5 * avg_volume_10)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    # Pre-compute 1w ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr_1w1 = high_1w - low_1w
    tr_1w2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr_1w3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr_1w1, np.maximum(tr_1w2, tr_1w3))
    tr_1w[0] = tr_1w1[0]
    atr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_filter = (atr_14 / close_1w) < 0.03  # ATR < 3% of price
    atr_filter_aligned = align_htf_to_ltf(prices, df_1w, atr_filter)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d ATR(20) for stoploss
    tr_1d1 = high_1d - low_1d
    tr_1d2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_1d3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))
    tr_1d[0] = tr_1d1[0]
    atr_20 = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_1d[i] < donchian_low[i] or close_1d[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_1d[i] > donchian_high[i] or close_1d[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume and volatility filters
            if vol_spike_aligned[i] and atr_filter_aligned[i]:
                # Long: price breaks above Donchian high
                if close_1d[i] > donchian_high[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close_1d[i] < donchian_low[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals