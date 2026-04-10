#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 12h chop regime filter
# - Long when Williams %R < -80 (oversold) AND 1d volume > 1.5x 20-period volume SMA AND 12h chop > 61.8 (ranging market)
# - Short when Williams %R > -20 (overbought) AND 1d volume > 1.5x 20-period volume SMA AND 12h chop > 61.8
# - Exit: Williams %R returns to -50 (mean reversion)
# - Uses 4h for price action (Williams %R), 1d for volume confirmation, 12h for chop filter
# - Target: 25-40 trades/year to minimize fee drag while capturing high-probability mean-reversion signals
# - Williams %R is effective in ranging markets; volume confirmation reduces false signals; chop filter ensures regime validity

name = "4h_1d_12h_williamsr_volspike_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 12h data ONCE before loop for chop filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 4h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    
    # Pre-compute 12h Chopiness Index (14-period) for regime filter
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    tr1 = np.abs(df_12h_high[1:] - df_12h_low[:-1])
    tr2 = np.abs(df_12h_high[1:] - df_12h_close[:-1])
    tr3 = np.abs(df_12h_low[1:] - df_12h_close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_12h_high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_12h_low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Chop filter: chop > 61.8 indicates ranging market (good for mean-reversion)
        chop_filter = chop_12h_aligned[i] > 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: Williams %R < -80 (oversold)
            if williams_r[i] < -80:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Williams %R > -20 (overbought)
            elif williams_r[i] > -20:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions: Williams %R returns to -50 (mean reversion)
            if position == 1 and williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals