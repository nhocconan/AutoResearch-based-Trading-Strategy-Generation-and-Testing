#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR trailing stop
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.3x 20-period volume SMA
# - Exit: ATR trailing stop (2.5 * ATR from extreme) or opposite Camarilla level (H3/L3)
# - Uses 4h for price action and Camarilla levels (derived from prior 1d OHLC), 1d for volume confirmation
# - Camarilla pivots work in ranging markets (mean reversion at H3/L3) and trending markets (breakouts)
# - Volume confirmation reduces false breakouts; ATR stop manages risk
# - Target: 20-40 trades/year to minimize fee drag while capturing high-probability moves

name = "4h_1d_camarilla_volbreak_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for pivot calculation and volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ATR for trailing stop (using 4h data)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track extreme prices for trailing stop
    long_high = np.full(n, np.nan)
    short_low = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after 20-bar warmup for volume SMA
        # Skip if any required data is invalid
        if np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Get prior completed 1d bar for Camarilla calculation (use previous day's OHLC)
        if i < 16:  # Need at least one 1d bar before current 4h bar (16x4h = 1d)
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from prior 1d bar
        # Each 4h bar = 1/4 of 1d bar, so prior 1d close is at index (i//16)*16 - 16
        prior_1d_start_idx = (i // 16) * 16 - 16
        prior_1d_end_idx = prior_1d_start_idx + 16
        
        # Ensure we don't go out of bounds
        if prior_1d_start_idx < 0 or prior_1d_end_idx > len(prices):
            signals[i] = 0.0
            continue
            
        # Get prior 1d OHLC from 4h data
        prior_high = np.max(high[prior_1d_start_idx:prior_1d_end_idx])
        prior_low = np.min(low[prior_1d_start_idx:prior_1d_end_idx])
        prior_close = close[prior_1d_end_idx - 1]  # Last 4h bar of prior day
        
        # Calculate Camarilla levels
        range_val = prior_high - prior_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
        # We use H3 and L3 for breakouts
        H3 = prior_close + range_val * 1.1 / 4
        L3 = prior_close - range_val * 1.1 / 4
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Only trade when volume confirmation is present
        if vol_confirm:
            # Update trailing stop extremes
            if position == 1:
                long_high[i] = max(long_high[i-1] if not np.isnan(long_high[i-1]) else close[i], close[i])
            elif position == -1:
                short_low[i] = min(short_low[i-1] if not np.isnan(short_low[i-1]) else close[i], close[i])
            else:
                long_high[i] = close[i]
                short_low[i] = close[i]
            
            # Long: price breaks above H3
            if close[i] > H3:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below L3
            elif close[i] < L3:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions
            exit_signal = False
            
            # Exit 1: ATR trailing stop (2.5 * ATR from extreme)
            if position == 1 and not np.isnan(long_high[i]):
                if close[i] < long_high[i] - 2.5 * atr[i]:
                    exit_signal = True
            elif position == -1 and not np.isnan(short_low[i]):
                if close[i] > short_low[i] + 2.5 * atr[i]:
                    exit_signal = True
            
            # Exit 2: Opposite Camarilla level (reversal signal)
            if not exit_signal:
                if position == 1 and close[i] < L3:
                    exit_signal = True
                elif position == -1 and close[i] > H3:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals