#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume spike and chop regime filter
# - Long when Williams %R(14) crosses above -80 (oversold reversal) AND 1d volume > 1.8x 20-period volume SMA AND chop > 61.8 (ranging market)
# - Short when Williams %R(14) crosses below -20 (overbought reversal) AND 1d volume > 1.8x 20-period volume SMA AND chop > 61.8
# - Exit: Williams %R crosses above -20 for longs or below -80 for shorts
# - Uses 4h for price action (Williams %R), 1d for volume confirmation, 4h for chop filter
# - Williams %R catches reversals in ranging markets; volume spike confirms participation; chop filter avoids trending markets where reversals fail
# - Tight entries target ~20-30 trades/year to minimize fee drag

name = "4h_1d_williamsr_volspike_chop_v1"
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
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 4h Williams %R (14-period)
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_14 - close) / (highest_14 - lowest_14)
    williams_r = np.where((highest_14 - lowest_14) == 0, -50, williams_r)  # avoid division by zero
    
    # Pre-compute 4h Chopiness Index (14-period) for regime filter
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    
    for i in range(14, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Chop filter: chop > 61.8 indicates ranging market (good for reversals)
        chop_filter = chop[i] > 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: Williams %R crosses above -80 (oversold reversal)
            if i > 0 and williams_r[i-1] <= -80 and williams_r[i] > -80:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Williams %R crosses below -20 (overbought reversal)
            elif i > 0 and williams_r[i-1] >= -20 and williams_r[i] < -20:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions: Williams %R crosses above -20 for longs or below -80 for shorts
            if position == 1 and williams_r[i] >= -20:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] <= -80:
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