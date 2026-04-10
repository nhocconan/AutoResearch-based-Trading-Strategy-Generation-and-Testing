#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar avg AND chop > 61.8 (ranging market)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar avg AND chop > 61.8
# - Exit when price returns to Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian breakouts capture volatility expansion; volume confirms institutional participation
# - Chop > 61.8 ensures we trade in ranging markets where mean reversion works best

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1d chop regime: > 61.8 = ranging (good for mean reversion)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes over 14 periods
    price_change = np.abs(np.diff(close_1d))
    sum_price_change = pd.Series(price_change).rolling(window=14, min_periods=14).sum().values
    sum_price_change = np.concatenate([[np.nan]*14, sum_price_change])
    
    # Choppiness Index: CHOP = 100 * log10(sum_price_change / (atr * 14)) / log10(14)
    chop_1d = 100 * np.log10(sum_price_change / (atr_1d * 14)) / np.log10(14)
    chop_1d = np.where((atr_1d * 14) > 0, chop_1d, 50)  # Avoid division by zero
    chop_high = chop_1d > 61.8  # Chop > 61.8 = ranging market
    chop_high_aligned = align_htf_to_ltf(prices, df_1d, chop_high)
    
    # Pre-compute Donchian channels from LTF data (4h)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_high_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND volume spike AND chop > 61.8 (ranging)
            if (prices['high'].iloc[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i] and 
                chop_high_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND volume spike AND chop > 61.8
            elif (prices['low'].iloc[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_high_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion)
            # Exit when price returns to Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['low'].iloc[i] <= donchian_mid[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['high'].iloc[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals