#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI + 1d Williams %R regime filter
# Volume-Weighted RSI (VW-RSI) incorporates volume into momentum, reducing false signals in low-volume moves
# 1d Williams %R identifies overbought/oversold extremes: > -20 = overbought, < -80 = oversold
# In trending markets (Williams %R between -80 and -20), VW-RSI crossovers signal continuation
# In extreme regimes (Williams %R <= -80 or >= -20), VW-RSI divergence from price signals reversals
# Volume confirmation (1.3x 20-period average) ensures participation
# Discrete position sizing 0.25 minimizes fee churn while maintaining meaningful exposure
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via Williams %R
# Uses 1d for HTF regime and VW-RSI calculation for stability

name = "6h_VolWeightedRSI_1dWilliamsR_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R regime and VW-RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d typical price for VW-RSI
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Volume-Weighted RSI (14-period)
    # Typical price change
    delta = np.diff(typical_price, prepend=typical_price[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_gains = gains * volume_1d
    vol_losses = losses * volume_1d
    
    # Smoothed volume-weighted gains and losses (Wilder's smoothing)
    avg_vol_gain = pd.Series(vol_gains).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_losses).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume-weighted RS and RSI
    rs = avg_vol_gain / (avg_vol_loss + 1e-10)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    
    # Align 1d indicators to 6h
    vw_rsi_aligned = align_htf_to_ltf(prices, df_1d, vw_rsi)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h volume confirmation (1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for VW-RSI, Williams %R and volume MA)
    start_idx = 50  # max(20 for volume, 34 for indicators) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(vw_rsi_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d Williams %R
        oversold = williams_r_aligned[i] <= -80
        overbought = williams_r_aligned[i] >= -20
        trending = (williams_r_aligned[i] > -80) and (williams_r_aligned[i] < -20)
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: VW-RSI crossovers signal continuation
                # Long: VW-RSI crosses above 50 from below
                if (vw_rsi_aligned[i] > 50 and 
                    i > start_idx and vw_rsi_aligned[i-1] <= 50 and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: VW-RSI crosses below 50 from above
                elif (vw_rsi_aligned[i] < 50 and 
                      i > start_idx and vw_rsi_aligned[i-1] >= 50 and
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif oversold:
                # In oversold regime: look for VW-RSI bullish divergence or bounce
                # Long: VW-RSI rises above 30 from below (bullish momentum returning)
                if (vw_rsi_aligned[i] > 30 and 
                    i > start_idx and vw_rsi_aligned[i-1] <= 30 and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif overbought:
                # In overbought regime: look for VW-RSI bearish divergence or fade
                # Short: VW-RSI falls below 70 from above (bearish momentum returning)
                if (vw_rsi_aligned[i] < 70 and 
                    i > start_idx and vw_rsi_aligned[i-1] >= 70 and
                    volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when VW-RSI falls below 50
                if vw_rsi_aligned[i] < 50:
                    exit_signal = True
            elif oversold:
                # Exit oversold long when VW-RSI rises above 70 (overbought bounce)
                if vw_rsi_aligned[i] > 70:
                    exit_signal = True
            else:  # overbought regime
                # Exit overbought long when VW-RSI falls below 30 (weakening bounce)
                if vw_rsi_aligned[i] < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when VW-RSI rises above 50
                if vw_rsi_aligned[i] > 50:
                    exit_signal = True
            elif oversold:
                # Exit oversold short when VW-RSI falls below 30 (weakening bounce)
                if vw_rsi_aligned[i] < 30:
                    exit_signal = True
            else:  # overbought regime
                # Exit overbought short when VW-RSI rises above 70 (overbought)
                if vw_rsi_aligned[i] > 70:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals