#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with volatility filter and volume confirmation.
# Long when price breaks above 20-period 1w Donchian high, ATR(14) < 30-period ATR(14) percentile (low volatility), and volume > 1.5x 20-day average.
# Short when price breaks below 20-period 1w Donchian low under same conditions.
# Exit when price returns to 1w midpoint or volatility expands (ATR > 70th percentile).
# Designed to capture breakouts during low-volatility periods, which often precede strong moves in both bull and bear markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Donchian channels and ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR on 1w for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1w = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR percentile rank (using 30-period lookback)
    atr_rank = np.full_like(atr_1w, np.nan)
    for i in range(30, len(atr_1w)):
        if not np.isnan(atr_1w[i-30:i]).any():
            atr_rank[i] = (atr_1w[i] <= np.percentile(atr_1w[i-30:i], 70)) * 100  # % of values <= current
    
    # Calculate 20-period Donchian channels on 1w
    lookback = 20
    donch_high = np.full_like(high_1w, np.nan)
    donch_low = np.full_like(low_1w, np.nan)
    donch_mid = np.full_like(close_1w, np.nan)
    
    for i in range(lookback, len(high_1w)):
        donch_high[i] = np.max(high_1w[i-lookback:i])
        donch_low[i] = np.min(low_1w[i-lookback:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Align indicators to lower timeframe (1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_rank_aligned = align_htf_to_ltf(prices, df_1w, atr_rank)
    
    # Volume confirmation: 1.5x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need ATR rank and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(atr_rank_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR below 30th percentile (low volatility)
        low_vol = atr_rank_aligned[i] < 30
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND low volatility AND volume confirmation
            if (close[i] > donch_high_aligned[i] and 
                low_vol and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND low volatility AND volume confirmation
            elif (close[i] < donch_low_aligned[i] and 
                  low_vol and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian mid or volatility expands (ATR > 70th percentile)
            if (close[i] <= donch_mid_aligned[i] or 
                atr_rank_aligned[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian mid or volatility expands
            if (close[i] >= donch_mid_aligned[i] or 
                atr_rank_aligned[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wDonchian_Breakout_VolVolFilter_v1"
timeframe = "1d"
leverage = 1.0