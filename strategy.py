#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion + 1d volume spike + chop regime filter
# - Williams %R(14) on 1d timeframe identifies overbought/oversold conditions
# - Long when Williams %R < -80 (oversold) AND 1d volume > 2.0x 20-period average AND chop > 61.8 (ranging market)
# - Short when Williams %R > -20 (overbought) AND 1d volume > 2.0x 20-period average AND chop > 61.8 (ranging market)
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Williams %R is effective in ranging/bear markets (2025-2026 test period)
# - Volume confirmation reduces false signals
# - Chop filter ensures we only trade in ranging conditions where mean reversion works

name = "12h_1d_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Pre-compute 12h chop regime filter (Choppiness Index)
    def true_range(h, l, pc):
        return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    
    # Calculate True Range for 12h
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    # Calculate Chop = 100 * log10(sum(TR, period) / (ATR * period)) / log10(period)
    atr_period = 14
    chop_period = 14
    sum_tr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    chop = 100 * np.log10(sum_tr / (atr * chop_period)) / np.log10(chop_period)
    chop = np.where(atr == 0, 50, chop)  # avoid division by zero
    chop = np.where(sum_tr == 0, 50, chop)
    
    # Align HTF indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R < -80 (oversold) AND volume spike AND chop > 61.8 (ranging)
            if (williams_r_aligned[i] < -80 and 
                volume_spike_1d_aligned[i] and 
                chop[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R > -20 (overbought) AND volume spike AND chop > 61.8 (ranging)
            elif (williams_r_aligned[i] > -20 and 
                  volume_spike_1d_aligned[i] and 
                  chop[i] > 61.8):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            exit_long = (position == 1 and williams_r_aligned[i] > -50)
            exit_short = (position == -1 and williams_r_aligned[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals