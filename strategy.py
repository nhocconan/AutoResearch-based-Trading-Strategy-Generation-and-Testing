#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly ATR volatility filter and volume confirmation
# - Long when price breaks above 20-period 6h Donchian upper band AND 1w ATR(14) > 1.5x 20-period 1w ATR SMA AND 1w volume > 1.3x 20-period 1w volume SMA
# - Short when price breaks below 20-period 6h Donchian lower band AND 1w ATR(14) > 1.5x 20-period 1w ATR SMA AND 1w volume > 1.3x 20-period 1w volume SMA
# - Exit: price returns to 20-period 6h Donchian midpoint or opposing breakout
# - Uses 1w for volatility and volume confirmation, 6h for price action and Donchian channels
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to avoid overtrading
# - Donchian channels identify volatility-based breakouts; volatility filter ensures we trade during expanded volatility periods
# - Volume confirmation ensures institutional participation; works in both bull and bear markets as breakouts occur in all regimes

name = "6h_1w_donchian_breakout_vol_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate 1w ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]  # First TR is just high-low
    
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_sma_20_1w = pd.Series(atr_14_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w ATR and ATR SMA to 6h timeframe
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_sma_20_1w)
    
    # Calculate 1w volume SMA for confirmation
    vol_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (highest_high_20 + lowest_low_20) / 2.0
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback for indicators
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr_14_1w_aligned[i]) or np.isnan(atr_sma_20_1w_aligned[i]) or 
            np.isnan(volume_sma_20_1w_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1w ATR(14) > 1.5x 20-period 1w ATR SMA
        vol_filter = atr_14_1w_aligned[i] > 1.5 * atr_sma_20_1w_aligned[i]
        
        # Volume confirmation: 1w volume > 1.3x 20-period 1w volume SMA
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
        vol_confirm = vol_1w_aligned[i] > 1.3 * volume_sma_20_1w_aligned[i]
        
        # Only trade when both volatility filter and volume confirmation are present
        if vol_filter and vol_confirm:
            # Long breakout: price breaks above 6h Donchian upper band
            if close[i] > highest_high_20[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below 6h Donchian lower band
            elif close[i] < lowest_low_20[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to 6h Donchian midpoint
            elif abs(close[i] - donchian_mid_20[i]) < (highest_high_20[i] - lowest_low_20[i]) * 0.05:
                if position != 0:  # Only signal on exit
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Maintain flat
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals