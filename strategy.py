#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR stoploss
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar average
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar average
# - Exit when price crosses Donchian(10) midpoint OR ATR-based stoploss hit
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation filters false breakouts
# - ATR stoploss manages risk during adverse moves
# - Works in ranging markets via mean-reversion exit at midpoint

name = "4h_1d_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian(20) for breakout signals
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for midpoint exit
    donchian_high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high_4h - low_4h).values
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1))).values
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-bar average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_20_avg)
    vol_spike = volume_1d > (1.5 * volume_20_avg)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian(20) high with volume spike
            if close_4h[i] > donchian_high_20[i] and vol_spike_aligned[i]:
                position = 1
                entry_price = close_4h[i]
                atr_stop = entry_price - 2.5 * atr[i]
                signals[i] = 0.25
            # Short signal: price breaks below Donchian(20) low with volume spike
            elif close_4h[i] < donchian_low_20[i] and vol_spike_aligned[i]:
                position = -1
                entry_price = close_4h[i]
                atr_stop = entry_price + 2.5 * atr[i]
                signals[i] = -0.25
        else:  # Have position - look for exit
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit 1: price crosses Donchian(10) midpoint (mean reversion)
                if close_4h[i] < donchian_mid_10[i]:
                    exit_signal = True
                # Exit 2: ATR stoploss hit
                elif close_4h[i] < atr_stop:
                    exit_signal = True
                # Exit 3: Donchian(20) breakdown (failed breakout)
                elif close_4h[i] < donchian_low_20[i]:
                    exit_signal = True
                    
            elif position == -1:  # Short position
                # Exit 1: price crosses Donchian(10) midpoint (mean reversion)
                if close_4h[i] > donchian_mid_10[i]:
                    exit_signal = True
                # Exit 2: ATR stoploss hit
                elif close_4h[i] > atr_stop:
                    exit_signal = True
                # Exit 3: Donchian(20) breakout (failed breakdown)
                elif close_4h[i] > donchian_high_20[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals