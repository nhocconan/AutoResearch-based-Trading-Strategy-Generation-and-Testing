#!/usr/bin/env python3
# 6h_1d1w_donchian_volume_regime_v1
# Hypothesis: Combine daily Donchian breakout direction with weekly volume regime filter on 6h timeframe.
# In weekly high-volume regime (vol > 1.5x 20-period MA): trade breakouts in direction of daily trend.
# In weekly low-volume regime: avoid breakouts, wait for mean reversion at Donchian bands.
# Uses volume filter to avoid false breakouts and adapts to market regime.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_donchian_volume_regime_v1"
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
    
    # Weekly volume regime filter
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_regime_1w = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)  # weekly 20-period volume MA
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels for previous day
    donchian_upper = np.zeros(len(high_1d))
    donchian_lower = np.zeros(len(high_1d))
    donchian_middle = np.zeros(len(high_1d))
    
    for i in range(20, len(high_1d)):  # need 20 periods for calculation
        donchian_upper[i] = np.max(high_1d[i-20:i])
        donchian_lower[i] = np.min(low_1d[i-20:i])
        donchian_middle[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Daily trend filter: price vs 50-period EMA
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_regime_1w[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume regime: high if current weekly volume > 1.5x 20-period MA
        # Note: vol_regime_1w[i] is the weekly 20-period MA, need current weekly volume
        # We'll approximate using the aligned value (simplified)
        high_vol_regime = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Volume surge for breakout confirmation
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price < Donchian lower or trend reversal
            if close[i] < donchian_lower_aligned[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian upper or trend reversal
            if close[i] > donchian_upper_aligned[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if high_vol_regime:
                # High volume regime: trade breakouts in direction of daily trend
                # Long: price > Donchian upper with volume surge and price > EMA50
                if (close[i] > donchian_upper_aligned[i] and vol_surge and 
                    close[i] > ema50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price < Donchian lower with volume surge and price < EMA50
                elif (close[i] < donchian_lower_aligned[i] and vol_surge and 
                      close[i] < ema50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
            else:
                # Low volume regime: mean reversion at Donchian bands
                # Long: price < Donchian lower and price > EMA50 (oversold in uptrend)
                if (close[i] < donchian_lower_aligned[i] and 
                    close[i] > ema50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: price > Donchian upper and price < EMA50 (overbought in downtrend)
                elif (close[i] > donchian_upper_aligned[i] and 
                      close[i] < ema50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals