#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 12h volume > 1.5x 20-period volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 12h volume > 1.5x 20-period volume SMA AND chop > 61.8
# - Exit: price crosses Camarilla pivot point (mean of high and low)
# - Uses 4h for price action (Camarilla levels), 12h for volume confirmation and chop filter
# - Camarilla pivots capture intraday support/resistance; volume confirms breakout strength; chop filter avoids strong trends
# - Target: 19-50 trades/year to minimize fee drag while capturing high-probability breakouts in ranging markets

name = "4h_12h_camarilla_volspike_chop_v1"
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
    
    # Load 12h data ONCE before loop for volume confirmation and chop filter (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Calculate 12h volume SMA for confirmation
    vol_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 4h Camarilla levels (based on previous bar's high, low, close)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    # Pivot = (high + low + close)/3
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_h3 = camarilla_pivot + 1.1 * camarilla_range / 4.0
    camarilla_l3 = camarilla_pivot - 1.1 * camarilla_range / 4.0
    
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
        if (np.isnan(camarilla_pivot[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = vol_12h_aligned[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Chop filter: chop > 61.8 indicates ranging market (good for breakout mean reversion)
        chop_filter = chop_12h_aligned[i] > 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: price breaks above Camarilla H3 level
            if close[i] > camarilla_h3[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Camarilla L3 level
            elif close[i] < camarilla_l3[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions: price crosses Camarilla pivot point
            if position == 1 and close[i] < camarilla_pivot[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > camarilla_pivot[i]:
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