#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-bar avg AND chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-bar avg AND chop < 61.8 (trending)
# - Exit when price touches Donchian(20) midpoint OR chop > 61.8 (range) OR volume < 1.5x 20-bar avg
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Donchian captures structural breaks; volume confirms institutional participation
# - Chop filter avoids whipsaws in ranging markets (proven edge in bear markets)
# - Target: 25-50 trades/year on 4h timeframe (100-200 total over 4 years)
# - Works in bull markets via breakouts and bear markets via short breakdowns

name = "4h_1d_donchian_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg_1d)
    vol_medium_1d = volume_1d > (1.5 * volume_20_avg_1d)  # for exit condition
    
    # Pre-compute 1d chop regime: < 61.8 = trending, > 61.8 = ranging
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: 100 * log10(sum(TR14)/ (max(high14)-min(low14))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    denominator = max_high_14 - min_low_14
    denominator = np.where(denominator == 0, 1e-10, denominator)  # Avoid division by zero
    
    chop_raw = 100 * np.log10(sum_tr_14 / denominator) / np.log10(14)
    chop = np.where(denominator <= 0, 50, chop_raw)  # Default to neutral when invalid
    chop = np.nan_to_num(chop, nan=50.0)
    
    chop_trending = chop < 61.8  # Trending regime
    chop_ranging = chop > 61.8   # Ranging regime
    
    # Align HTF indicators to 4h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    vol_medium_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_medium_1d)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    
    # Pre-compute Donchian channels on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20): 20-period high/low
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2.0
    
    # Breakout conditions
    breakout_up = high > donch_high_20  # Price breaks above upper band
    breakout_down = low < donch_low_20  # Price breaks below lower band
    
    # Exit conditions: touch midpoint OR chop ranges OR volume drops
    exit_mid = np.abs(close - donch_mid_20) < 0.001 * close  # Within 0.1% of midpoint
    exit_chop_range = chop_ranging_aligned  # Chop > 61.8 = ranging
    exit_vol_drop = ~vol_medium_1d_aligned  # Volume < 1.5x avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_spike_1d_aligned[i]) or np.isnan(vol_medium_1d_aligned[i]) or
            np.isnan(chop_trending_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or
            np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or np.isnan(donch_mid_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when breakout up AND volume spike AND trending chop
            if (breakout_up[i] and 
                vol_spike_1d_aligned[i] and 
                chop_trending_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short when breakout down AND volume spike AND trending chop
            elif (breakout_down[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_trending_aligned[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit when: price touches midpoint OR chop ranges OR volume drops significantly
            exit_signal = (exit_mid[i] or 
                          exit_chop_range[i] or 
                          exit_vol_drop[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals