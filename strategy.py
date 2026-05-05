#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and ATR-based position sizing
# Long when price breaks above weekly Donchian high(20) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below weekly Donchian low(20) AND volume > 1.5 * avg_volume(20)
# Position size scaled by ATR volatility (0.20 in low vol, 0.35 in high vol) to adapt to market conditions
# Exit when price crosses back below/above weekly Donchian midpoint
# Weekly Donchian provides robust structure from higher timeframe, reducing false breakouts
# Volume confirmation ensures breakout strength, ATR scaling manages risk across regimes
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume)

name = "1d_Donchian20_Weekly_VolumeSpike_ATRsize"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian(20) channels (based on completed weekly bars)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2.0  # Midpoint for exit
    
    # Align weekly Donchian levels to 1d timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate ATR(14) for volatility-based position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalize ATR to [0,1] range over 50-period for adaptive sizing
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=10).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    atr_ratio = np.clip(atr_ratio, 0.5, 2.0)  # Bound between 0.5x and 2x average
    
    # Base size 0.25 scaled by ATR ratio (0.20 in low vol, 0.35 in high vol)
    base_size = 0.25
    size_multiplier = 0.6 + 0.6 * (atr_ratio - 0.5) / 1.5  # Maps 0.5→0.6, 2.0→1.2
    position_size = base_size * size_multiplier
    position_size = np.clip(position_size, 0.20, 0.35)  # Final size bounds
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high AND volume confirmation
            if close[i] > donchian_high_aligned[i] and volume_confirm[i]:
                signals[i] = position_size[i]
                position = 1
            # Short: Price breaks below weekly Donchian low AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and volume_confirm[i]:
                signals[i] = -position_size[i]
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals