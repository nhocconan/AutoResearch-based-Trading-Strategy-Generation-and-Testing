#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period 1d volume SMA AND chop > 61.8 (range)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period 1d volume SMA AND chop > 61.8 (range)
# - Exit: ATR trailing stop (2.5x ATR from extreme)
# - Uses 1d for volume confirmation, 4h for Camarilla levels and ATR, chop filter from 4h
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 15-30 trades/year (60-120 total over 4 years) to avoid overtrading
# - Camarilla pivots work well in ranging markets; chop filter ensures we only trade in ranging conditions
# - Volume spike confirms institutional interest in the breakout

name = "4h_1d_camarilla_pivot_volume_chop_v1"
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
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h ATR for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Camarilla pivot levels (based on previous day's OHLC)
    # We need to calculate these from 1d data and align to 4h
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # But we need to use previous day's OHLC for today's levels
    df_1d = df_1d.copy()
    df_1d['prev_close'] = df_1d['close'].shift(1)
    df_1d['prev_high'] = df_1d['high'].shift(1)
    df_1d['prev_low'] = df_1d['low'].shift(1)
    
    # Calculate Camarilla levels using previous day's data
    df_1d['cam_h3'] = df_1d['prev_close'] + 1.1 * (df_1d['prev_high'] - df_1d['prev_low']) / 2
    df_1d['cam_l3'] = df_1d['prev_close'] - 1.1 * (df_1d['prev_high'] - df_1d['prev_low']) / 2
    
    # Align Camarilla levels to 4h timeframe
    cam_h3_1d = df_1d['cam_h3'].values
    cam_l3_1d = df_1d['cam_l3'].values
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3_1d)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3_1d)
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(atr over period) / (log10(highest-high - lowest-low) * log10(period)))
    # Simplified: CHOP = 100 * log10(sum(atr14) / (log10(max(high) - min(low)) * log10(14)))
    # We'll use a common approximation: CHOP = 100 * log10(sum(atr14,14) / (log10(highest-high14) * log10(14)))
    # Actually, standard formula: CHOP = 100 * LOG10(SUM(ATR1, n) / (LOG10(HIGHMAX - LOWMIN) * LOG10(n)))
    # We'll implement correctly:
    atr_1 = tr  # True Range
    sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero or log of zero
    hl_range = highest_high_14 - lowest_low_14
    hl_range = np.where(hl_range <= 0, 1e-10, hl_range)  # small positive value
    chop = 100 * np.log10(sum_atr_14 / (np.log10(hl_range) * np.log10(14)))
    # CHOP values typically between 0-100; >61.8 = ranging, <38.2 = trending
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(cam_h3_aligned[i]) or np.isnan(cam_l3_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (need to align properly)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        # Volume confirmation: 1d volume > 2.0x 20-period 1d volume SMA
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop[i] > 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > cam_h3_aligned[i-1]  # Break above H3 level
        breakout_down = close[i] < cam_l3_aligned[i-1]  # Break below L3 level
        
        if position == 0:  # Flat - look for entry
            # Require both volume confirmation and chop filter
            if vol_confirm and chop_filter:
                # Long: price breaks above Camarilla H3
                if breakout_up:
                    position = 1
                    signals[i] = 0.25
                    highest_since_entry[i] = high[i]  # Initialize trailing stop
                # Short: price breaks below Camarilla L3
                elif breakout_down:
                    position = -1
                    signals[i] = -0.25
                    lowest_since_entry[i] = low[i]  # Initialize trailing stop
                else:
                    signals[i] = 0.0
                    # Carry forward NaN values for tracking
                    if i > 0:
                        highest_since_entry[i] = highest_since_entry[i-1]
                        lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 2.5*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.5 * atr[i]
            
            # Exit condition: trailing stop hit
            if close[i] < trailing_stop:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 2.5*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.5 * atr[i]
            
            # Exit condition: trailing stop hit
            if close[i] > trailing_stop:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals