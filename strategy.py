#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian Breakout + Volume Confirmation
# Uses Bollinger Band Width (BBW) on 6h to detect regime: low BBW = squeeze (breakout imminent),
# high BBW = expansion (trend or chop). In low BBW regime, trade 1d Donchian breakouts.
# In high BBW regime, fade 6h moves toward VWAP. Volume confirmation ensures participation.
# Works in bull/bear by adapting to volatility regime via BBW.
# Targets 12-37 trades/year (50-150 total over 4 years) with discrete sizing 0.25.

name = "6h_BBWRegime_1dDonchian_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donch_high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 6h Bollinger Band Width (20, 2)
    ma_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_6h + 2 * std_6h
    lower_bb = ma_6h - 2 * std_6h
    bb_width = (upper_bb - lower_bb) / ma_6h
    
    # Calculate 6h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_num / (vwap_den + 1e-10)
    
    # Calculate 6h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Bollinger Band Width regime thresholds
    bbw_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bbw_std = pd.Series(bb_width).rolling(window=50, min_periods=50).std().values
    bbw_lower = bbw_ma - bbw_std  # Low BBW = squeeze
    bbw_upper = bbw_ma + bbw_std  # High BBW = expansion
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for BBW, Donchian, VWAP)
    start_idx = 70  # max(50 for BBW regime, 20 for Donchian/VWAP) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(bb_width[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(bbw_lower[i]) or 
            np.isnan(bbw_upper[i])):
            signals[i] = 0.0
            continue
        
        # Regime detection via Bollinger Band Width
        squeeze = bb_width[i] < bbw_lower[i]  # Low volatility, breakout imminent
        expansion = bb_width[i] > bbw_upper[i]  # High volatility, trend/chop
        
        if position == 0:  # Flat - look for new entries
            if squeeze:
                # In squeeze regime: trade 1d Donchian breakouts
                # Long: price breaks above 1d Donchian high
                if close[i] > donch_high_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d Donchian low
                elif close[i] < donch_low_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # expansion or neutral regime
                # In expansion regime: fade 6h moves toward VWAP (mean reversion)
                # Long: price below VWAP and moving up
                if close[i] < vwap[i] and i > start_idx and close[i-1] <= close[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price above VWAP and moving down
                elif close[i] > vwap[i] and i > start_idx and close[i-1] >= close[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if squeeze:
                # Exit squeeze long when price returns to 6h VWAP (breakout failed)
                if close[i] >= vwap[i]:
                    exit_signal = True
                # Or if Donchian breakout reverses
                elif close[i] < donch_low_aligned[i]:
                    exit_signal = True
            else:
                # Exit expansion long when price crosses above VWAP (mean reversion complete)
                if close[i] >= vwap[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if squeeze:
                # Exit squeeze short when price returns to 6h VWAP (breakdown failed)
                if close[i] <= vwap[i]:
                    exit_signal = True
                # Or if Donchian breakout reverses
                elif close[i] > donch_high_aligned[i]:
                    exit_signal = True
            else:
                # Exit expansion short when price crosses below VWAP (mean reversion complete)
                if close[i] <= vwap[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals