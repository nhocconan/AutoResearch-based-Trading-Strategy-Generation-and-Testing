#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ATR(14) stoploss
# - Primary signal: Price breaks above/below 20-period 12h Donchian channel
# - Volume confirmation: 12h volume > 1.5x 20-period median volume (avoid low-participation breakouts)
# - Trend filter: Price must be above/below 1d EMA50 for alignment with higher timeframe trend
# - Exit: ATR-based trailing stop (3*ATR from extreme) or opposite Donchian breakout
# - Position size: 0.25 (discrete level) to balance return and drawdown
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, volume filter reduces false signals, ATR stop manages risk in volatile markets

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian upper/lower bands
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 12h volume regime: volume > 1.5x 20-period median volume
    volume_12h = prices['volume'].values
    median_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    volume_regime = volume_12h > (1.5 * median_volume_20)
    
    # 12h ATR(14) for stoploss calculation
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_regime[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update long stop (trailing)
            long_stop = max(long_stop, high_12h[i] - 3.0 * atr_14[i])
            
            # Exit conditions: stop hit OR price breaks below Donchian lower
            if low_12h[i] <= long_stop or close_12h[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update short stop (trailing)
            short_stop = min(short_stop, low_12h[i] + 3.0 * atr_14[i])
            
            # Exit conditions: stop hit OR price breaks above Donchian upper
            if high_12h[i] >= short_stop or close_12h[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and 1d EMA50 filter
            # Long: Price breaks above Donchian upper AND volume regime AND price above 1d EMA50
            if (close_12h[i] > donchian_upper[i] and 
                volume_regime[i] and 
                close_12h[i] > ema_50_aligned[i]):
                position = 1
                entry_price = close_12h[i]
                long_stop = entry_price - 3.0 * atr_14[i]
                signals[i] = 0.25
            # Short: Price breaks below Donchian lower AND volume regime AND price below 1d EMA50
            elif (close_12h[i] < donchian_lower[i] and 
                  volume_regime[i] and 
                  close_12h[i] < ema_50_aligned[i]):
                position = -1
                entry_price = close_12h[i]
                short_stop = entry_price + 3.0 * atr_14[i]
                signals[i] = -0.25
    
    return signals