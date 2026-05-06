#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1d Donchian upper (20-period) AND volume > 1.5 * avg_volume(20) AND CHOP(14) > 61.8 (range regime)
# Short when price breaks below 1d Donchian lower (20-period) AND volume > 1.5 * avg_volume(20) AND CHOP(14) > 61.8 (range regime)
# Exit when price crosses 1d EMA34 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian provides clear structure with proven breakout edge in ranging markets
# Volume confirmation filters weak breakouts (reduces false signals)
# Choppiness filter ensures we only trade in range regimes where mean reversion works
# Works in bull (breakouts above upper in ranging markets) and bear (breakdowns below lower in ranging markets)

name = "12h_1dDonchian20_VolumeChop_EMA34Exit_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels, EMA, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34 and choppiness
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period) based on previous 1d bar
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for exit signal
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low))
    tr1 = pd.Series(high_1d).rolling(window=14).max() - pd.Series(low_1d).rolling(window=14).min()
    tr2 = abs(pd.Series(high_1d).rolling(window=14).max() - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).rolling(window=14).min() - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_14.rolling(window=14, min_periods=14).sum() / np.log10(14)) / np.log10((highest_high_14 - lowest_low_14).replace(0, np.nan))
    chop_values = chop.values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume confirmation and chop > 61.8 (range regime)
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume confirmation and chop > 61.8 (range regime)
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals