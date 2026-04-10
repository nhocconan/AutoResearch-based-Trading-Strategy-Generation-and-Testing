#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above 20-period Donchian high AND 1d volume > 2.0x 20-period volume SMA AND 1d Choppiness Index > 61.8 (ranging market)
# - Short when price breaks below 20-period Donchian low AND 1d volume > 2.0x 20-period volume SMA AND 1d Choppiness Index > 61.8 (ranging market)
# - Exit: ATR trailing stop (2.0 * ATR from extreme) or opposite Donchian breakout
# - Uses 4h for price action and Donchian channels, 1d for volume and choppiness filters
# - Choppiness filter ensures we trade only in ranging markets where mean reversion at channel extremes works
# - Volume spike adds conviction to breakouts (panic or euphoria)
# - ATR trailing stop manages risk while letting winners run
# - Target: 20-35 trades/year to minimize fee drag while capturing high-probability mean reversion

name = "4h_1d_donchian_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for volume and choppiness confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_1d = hh_1d - ll_1d
    chop = 100 * np.log10(tr_sum_14 / hh_ll_1d) / np.log10(14)
    chop = np.where(hh_ll_1d == 0, 50, chop)  # Avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute Donchian channels for 4h data (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA (volume spike)
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: 1d Choppiness Index > 61.8 (ranging market)
        regime_filter = chop_aligned[i] > 61.8
        
        # Only trade when both volume confirmation and regime filter are present
        if vol_confirm and regime_filter:
            # Update trailing stop extremes
            if position == 1:
                long_high[i] = max(long_high[i-1] if not np.isnan(long_high[i-1]) else close[i], close[i])
            elif position == -1:
                short_low[i] = min(short_low[i-1] if not np.isnan(short_low[i-1]) else close[i], close[i])
            else:
                long_high[i] = close[i]
                short_low[i] = close[i]
            
            # Long: price breaks above Donchian high
            if close[i] > donch_high[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Donchian low
            elif close[i] < donch_low[i]:
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
            
            # Exit 1: ATR trailing stop (2.0 * ATR from extreme)
            if position == 1 and not np.isnan(long_high[i]):
                if close[i] < long_high[i] - 2.0 * atr[i]:
                    exit_signal = True
            elif position == -1 and not np.isnan(short_low[i]):
                if close[i] > short_low[i] + 2.0 * atr[i]:
                    exit_signal = True
            
            # Exit 2: Opposite Donchian breakout (reversal signal)
            if not exit_signal:
                if position == 1 and close[i] < donch_low[i]:
                    exit_signal = True
                elif position == -1 and close[i] > donch_high[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
        else:
            # No volume or regime confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals