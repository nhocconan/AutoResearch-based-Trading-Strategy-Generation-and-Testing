#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + Donchian Breakout with 1d Volume Spike
# Long when: BBW < 20th percentile (low volatility squeeze) AND price breaks above Donchian(20) high AND 1d volume > 1.5x 20-day avg
# Short when: BBW < 20th percentile (low volatility squeeze) AND price breaks below Donchian(20) low AND 1d volume > 1.5x 20-day avg
# Exit when price returns to Donchian midpoint (mean reversion) OR BBW > 50th percentile (volatility expansion)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 75-150 total trades over 4 years (19-37/year) on 6h.
# Bollinger Band Width identifies low-volatility regimes ripe for breakouts.
# Donchian channels provide objective breakout levels.
# 1d volume filter ensures institutional participation.
# Works in both bull and bear markets as it trades volatility contractions/expansions, not direction.

name = "6h_BBW_Regime_Donchian_Breakout_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Band Width (20,2) - regime filter
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20  # Normalized width
    
    # Percentile rank of BBW (20-bar lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Donchian Channel (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 20-day volume average on 1d
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Volume spike: current 1d volume > 1.5x 20-day average
    volume_spike = volume_1d_aligned > 1.5 * volume_ma_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for BBW percentile (100-bar lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        bbw_percentile = bb_width_percentile[i]
        vol_spike = volume_spike[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        donch_high_i = donch_high[i]
        donch_low_i = donch_low[i]
        donch_mid_i = donch_mid[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint OR BBW > 50th percentile (vol expansion)
            if curr_close <= donch_mid_i or bbw_percentile > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint OR BBW > 50th percentile (vol expansion)
            if curr_close >= donch_mid_i or bbw_percentile > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Low volatility regime: BBW < 20th percentile (squeeze)
            low_vol_regime = bbw_percentile < 20
            
            # Long when: low vol AND price breaks above Donchian high AND 1d volume spike
            if low_vol_regime and curr_high > donch_high_i and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when: low vol AND price breaks below Donchian low AND 1d volume spike
            elif low_vol_regime and curr_low < donch_low_i and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals