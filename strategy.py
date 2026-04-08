#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 12h Trend and Volume Confirmation
# Uses Williams Alligator (three smoothed moving averages: Jaw, Teeth, Lips) on 6h timeframe
# to identify trend direction and potential reversals. Trades only when 12h trend is strong (ADX > 25)
# with volume confirmation (>1.5x 20-period average). In strong trends, the Alligator's
# mouth opens (JAW < TEETH < LIPS for uptrend, JAW > TEETH > LIPS for downtrend), signaling
# entry. Exit when the mouth closes (lines intertwine) or trend weakens.
# Target: 12-37 trades/year via Alligator + trend + volume confluence.
name = "6h_alligator_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:  # Need sufficient data for Alligator smoothing
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period ADX for trend strength on 12h data
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator components on 6h data (using close prices)
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # Shift forward 8 periods
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # Shift forward 5 periods
    teeth[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: 5-period SMMA, shifted 3 bars ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # Shift forward 3 periods
    lips[:3] = np.nan  # First 3 values invalid after shift
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback (need to account for Alligator shifts)
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h values for current 6h bar
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Alligator signals
        # Uptrend: JAW < TEETH < LIPS (mouth opening upward)
        alligator_long = jaw[i] < teeth[i] and teeth[i] < lips[i]
        # Downtrend: JAW > TEETH > LIPS (mouth opening downward)
        alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
        # Market sleeping (no trend): lines intertwined
        # (No explicit check needed - handled by else conditions)
        
        if position == 1:  # Long position
            # Exit: Alligator closes mouth OR trend weakens
            if not alligator_long or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator closes mouth OR trend weakens
            if not alligator_short or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade during strong trend with volume confirmation
            if strong_trend and volume_confirm:
                # Long: Alligator mouth opens upward
                if alligator_long:
                    position = 1
                    signals[i] = 0.25
                # Short: Alligator mouth opens downward
                elif alligator_short:
                    position = -1
                    signals[i] = -0.25
    
    return signals