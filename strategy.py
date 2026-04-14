#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation
# Uses Choppiness Index (14) to determine market regime: >61.8 = range (mean reversion), <38.2 = trending
# In trending regime: trade 1d Donchian(20) breakout in direction of trend
# In ranging regime: trade mean reversion at Bollinger Bands (20,2) with 1d trend filter
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to adapt to both trending and ranging markets, reducing whipsaw in chop
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channels (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA (50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Choppiness Index (14) - using 1d data
    atr_1d = pd.Series(np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(df_1d['close'], 1))), np.abs(low_1d - np.roll(df_1d['close'], 1)))).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (max_hh - min_ll)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Bollinger Bands (20, 2) for ranging market
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 1d EMA and Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ma_20[i]) or np.isnan(std_20[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Trending regime (CHOP < 38.2): Donchian breakout
            if chop_val < 38.2:
                # Long: price breaks above 1d Donchian high with volume filter and uptrend
                if price > donchian_high_aligned[i] and vol > 1.5 * avg_vol[i] and price > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below 1d Donchian low with volume filter and downtrend
                elif price < donchian_low_aligned[i] and vol > 1.5 * avg_vol[i] and price < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            # Ranging regime (CHOP > 61.8): Mean reversion at Bollinger Bands
            elif chop_val > 61.8:
                # Long: price touches lower BB with volume filter and 1d uptrend
                if price <= lower_bb[i] and vol > 1.5 * avg_vol[i] and price > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: price touches upper BB with volume filter and 1d downtrend
                elif price >= upper_bb[i] and vol > 1.5 * avg_vol[i] and price < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            # Neutral regime (38.2 <= CHOP <= 61.8): no trading
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions
            if chop_val < 38.2:  # trending: exit when price crosses 1d EMA
                if price < ema_50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # ranging or neutral: exit when price reaches middle Bollinger Band
                if price >= ma_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
        elif position == -1:
            # Exit conditions
            if chop_val < 38.2:  # trending: exit when price crosses 1d EMA
                if price > ema_50_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # ranging or neutral: exit when price reaches middle Bollinger Band
                if price <= ma_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
    
    return signals

name = "4h_Chop_DonchianBB_Regime"
timeframe = "4h"
leverage = 1.0