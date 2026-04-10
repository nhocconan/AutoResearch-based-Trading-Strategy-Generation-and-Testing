#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above 20-period 4h Donchian upper band AND 1d ATR(14) > 1.5x 20-period 1d ATR SMA AND 1d volume > 1.3x 20-period 1d volume SMA
# - Short when price breaks below 20-period 4h Donchian lower band AND 1d ATR(14) > 1.5x 20-period 1d ATR SMA AND 1d volume > 1.3x 20-period 1d volume SMA
# - Exit: price returns to 20-period 4h Donchian midpoint or opposing breakout
# - Uses 1d for volatility and volume confirmation, 4h for price action and Donchian channels
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to avoid overtrading
# - Donchian channels identify volatility-based breakouts; volatility filter ensures we trade during expanded volatility periods
# - Volume confirmation ensures institutional participation; works in both bull and bear markets as breakouts occur in all regimes

name = "4h_1d_donchian_breakout_vol_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First TR is just high-low
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_sma_20_1d = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR SMA to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20_1d)
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h Donchian channels (20-period)
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
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_sma_20_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR(14) > 1.5x 20-period 1d ATR SMA
        vol_filter = atr_14_1d_aligned[i] > 1.5 * atr_sma_20_1d_aligned[i]
        
        # Volume confirmation: 1d volume > 1.3x 20-period 1d volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Only trade when both volatility filter and volume confirmation are present
        if vol_filter and vol_confirm:
            # Long breakout: price breaks above 4h Donchian upper band
            if close[i] > highest_high_20[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Maintain position
            # Short breakout: price breaks below 4h Donchian lower band
            elif close[i] < lowest_low_20[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Maintain position
            # Exit: price returns to 4h Donchian midpoint
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