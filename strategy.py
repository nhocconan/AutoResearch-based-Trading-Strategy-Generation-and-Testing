#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian(20) Breakout
# Uses BB Width percentile to detect range (CHOP) vs trend regimes on 6h.
# In range (BBW > 60th percentile): fade at 1d Donchian bands with volume confirmation.
# In trend (BBW < 40th percentile): breakout continuation in direction of 1d Donchian.
# Volume spike (>1.5x 20-bar MA) confirms institutional participation.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.

name = "6h_BBWRegime_1dDonchian20_BreakoutFade_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands (20, 2) for regime detection
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # Normalized width
    
    # BB Width percentile rank (lookback 50 periods) for regime
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # 1d HTF data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Donchian(20) channels
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for BB width rank
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width_rank[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(bb_mid[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Regime detection: BB Width percentile
        bbw_rank = bb_width_rank[i]
        is_range = bbw_rank > 0.60   # High BB Width = ranging/Chop
        is_trend = bbw_rank < 0.40   # Low BB Width = trending
        
        if position == 0:  # Flat - look for new entries
            # RANGE REGIME: Fade at 1d Donchian bands
            if is_range and vol_spike:
                # Long: price at or below 1d Donchian low (support)
                if curr_low <= donch_low_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price at or above 1d Donchian high (resistance)
                elif curr_high >= donch_high_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # TREND REGIME: Breakout continuation in direction of 1d Donchian
            elif is_trend and vol_spike:
                # Long: breakout above 1d Donchian high
                if curr_high > donch_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: breakdown below 1d Donchian low
                elif curr_low < donch_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on return to 1d Donchian midpoint (mean reversion) or opposite band touch
            bb_mid_1d = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
            if curr_close >= bb_mid_1d or curr_low <= donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on return to 1d Donchian midpoint or opposite band touch
            bb_mid_1d = (donch_high_aligned[i] + donch_low_aligned[i]) / 2
            if curr_close <= bb_mid_1d or curr_high >= donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals