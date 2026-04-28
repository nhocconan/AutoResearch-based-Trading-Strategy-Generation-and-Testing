#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian Breakout + Volume Spike
# Bollinger Band Width < 20th percentile = low volatility squeeze (range regime)
# Donchian(20) breakout on 1d with volume > 2x 20-bar average = expansion signal
# Long when 1d price breaks above Donchian upper band AND 6h BBW < 20th percentile AND volume spike
# Short when 1d price breaks below Donchian lower band AND 6h BBW < 20th percentile AND volume spike
# BBW regime filter ensures we only trade breakouts after low volatility periods, reducing false breakouts
# Works in both bull and bear markets by capturing expansion after contraction
# Target: 12-30 trades/year via tight regime + breakout + volume confluence

name = "6h_BBW_Regime_1dDonchian20_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    
    # Calculate Donchian(20) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Prepend zeros for alignment (since we lost first 19 bars in calculations)
    donch_high_20 = np.concatenate([np.full(19, np.nan), donch_high_20])
    donch_low_20 = np.concatenate([np.full(19, np.nan), donch_low_20])
    
    # Align 1d Donchian bands to 6h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate Bollinger Band Width on 6h data (20, 2)
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # Percentage width
    
    # Calculate 20th percentile of BBW for regime filter (using 100-bar lookback)
    bbw_series = pd.Series(bb_width)
    bbw_percentile_20 = bbw_series.rolling(window=100, min_periods=100).quantile(0.20).values
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(120, 20)  # Need sufficient history for all indicators (100 for BBW percentile + 20 for others)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(bbw_percentile_20[i]) or np.isnan(sma_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        bbw_regime = bb_width[i] < bbw_percentile_20[i]  # Low volatility squeeze regime
        close_price = close[i]
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND BBW regime (low vol) AND volume spike
            if close_price > donch_high and bbw_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND BBW regime (low vol) AND volume spike
            elif close_price < donch_low and bbw_regime and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to middle of Donchian channel or volatility expands
            donch_mid = (donch_high + donch_low) / 2
            if close_price < donch_mid or not bbw_regime:  # Exit when price retracs to mid or volatility expands
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to middle of Donchian channel or volatility expands
            donch_mid = (donch_high + donch_low) / 2
            if close_price > donch_mid or not bbw_regime:  # Exit when price retracs to mid or volatility expands
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals