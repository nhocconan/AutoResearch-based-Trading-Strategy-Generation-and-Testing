#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w ADX trend filter and volume confirmation
# - Donchian(20) from 1d: upper/lower bands for breakout detection
# - 1w ADX(14) > 25 to ensure strong weekly trend and avoid choppy markets
# - Volume confirmation: current 1d volume > 1.5x 20-period average to confirm institutional participation
# - Designed for 1d timeframe: targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: weekly ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.30) to minimize fee churn while maintaining adequate exposure

name = "1d_1w_donchian_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(25) for trend filter (higher threshold for stronger trend)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values with longer period for stability
    tr_25 = pd.Series(tr).ewm(span=25, adjust=False, min_periods=25).mean().values
    dm_plus_25 = pd.Series(dm_plus).ewm(span=25, adjust=False, min_periods=25).mean().values
    dm_minus_25 = pd.Series(dm_minus).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_25 / tr_25
    di_minus = 100 * dm_minus_25 / tr_25
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=25, adjust=False, min_periods=25).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (trend reversal)
            if prices['close'].iloc[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (trend reversal)
            if prices['close'].iloc[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above Donchian upper band
                if prices['close'].iloc[i] > donchian_upper[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.30
                # Breakout short: price closes below Donchian lower band
                elif prices['close'].iloc[i] < donchian_lower[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.30
    
    return signals