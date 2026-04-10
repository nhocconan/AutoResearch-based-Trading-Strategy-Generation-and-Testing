#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.8x 20-bar avg AND chop(14) < 38.2 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.8x 20-bar avg AND chop(14) < 38.2 (trending)
# - Exit when price touches Donchian(20) midpoint OR chop(14) > 61.8 (range regime)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Donchian breakouts capture strong trending moves; volume confirms institutional participation
# - Choppiness filter avoids whipsaws in ranging markets (critical for 2022 bear and 2025+ range)
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Works in bull markets via breakouts; avoids bear market losses via regime filter and flat positioning

name = "12h_1d_donchian_volume_chop_regime_v1"
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
    
    # Pre-compute 1d ATR(14) for choppiness calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR(14) over last 14 periods
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max High - Min Low over last 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: CHOP = 100 * log10(atr_sum_14 / range_14) / log10(14)
    # Avoid division by zero and invalid values
    chop_ratio = np.where(range_14 > 0, atr_sum_14 / range_14, 1.0)
    chop_ratio = np.maximum(chop_ratio, 1e-10)  # Prevent log(0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Regime filters: chop < 38.2 = trending, chop > 61.8 = ranging
    chop_trending = chop < 38.2
    chop_ranging = chop > 61.8
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg_1d)
    
    # Align HTF indicators to 12h timeframe
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Donchian(20) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Breakout conditions
    breakout_up = high > donchian_high  # Current high breaks above Donchian high
    breakout_down = low < donchian_low  # Current low breaks below Donchian low
    
    # Session filter: 00-23 UTC (12h timeframe trades infrequently, no need to restrict)
    # But we'll use a mild filter to avoid extreme low-volume hours if needed
    hours = prices.index.hour
    in_session = ((hours >= 0) & (hours <= 23))  # Always true, kept for structure
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(chop_trending_aligned[i]) or
            np.isnan(chop_ranging_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter (always true but kept for consistency)
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike AND trending regime
            if (breakout_up[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND trending regime
            elif (breakout_down[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when price touches Donchian midpoint OR chop indicates ranging regime
            exit_signal = (np.abs(close[i] - donchian_mid[i]) < 0.001 * close[i]) or chop_ranging_aligned[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals