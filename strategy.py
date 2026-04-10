#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume spike + chop regime filter
# - Primary: 12h Camarilla pivot levels (based on prior 1d candle) for directional bias
# - HTF: 1d volume confirmation (current day volume > 1.5x 20-day MA) + chop regime filter (CHOP < 50 = trending)
# - Long: Price breaks above Camarilla H3 + volume confirmation + chop regime (trending)
# - Short: Price breaks below Camarilla L3 + volume confirmation + chop regime (trending)
# - Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or chop regime shifts to ranging (CHOP > 60)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla pivots capture institutional levels, volume confirms conviction, chop filter avoids false signals in ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    chop_lookback = 14
    tr1 = np.maximum(np.maximum(high_1d - low_1d,
                               np.abs(np.roll(high_1d, 1) - low_1d)),
                    np.abs(np.roll(low_1d, 1) - high_1d))
    
    # Sum of TR over chop_lookback period
    sum_tr = np.full(len(tr1), np.nan)
    for i in range(chop_lookback, len(tr1)):
        if not np.isnan(tr1[i-chop_lookback:i]).any():
            sum_tr[i] = np.sum(tr1[i-chop_lookback:i])
    
    # Highest high and lowest low over chop_lookback period
    hh = np.full(len(high_1d), np.nan)
    ll = np.full(len(low_1d), np.nan)
    for i in range(chop_lookback, len(high_1d)):
        if not np.isnan(high_1d[i-chop_lookback:i+1]).any() and not np.isnan(low_1d[i-chop_lookback:i+1]).any():
            hh[i] = np.max(high_1d[i-chop_lookback:i+1])
            ll[i] = np.min(low_1d[i-chop_lookback:i+1])
    
    # Chopiness Index
    chop = np.full(len(high_1d), np.nan)
    for i in range(chop_lookback, len(high_1d)):
        if (not np.isnan(sum_tr[i]) and not np.isnan(hh[i]) and not np.isnan(ll[i]) and 
            hh[i] > ll[i] and sum_tr[i] > 0):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(chop_lookback)
        else:
            chop[i] = np.nan
    
    # Align all HTF indicators to 12h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar to have previous day for pivot calc
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 12h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP < 50 indicates trending market (avoid ranging)
        regime_confirm = chop_aligned[i] < 50.0
        
        # Calculate Camarilla pivot levels from previous 1d candle
        # Only calculate at the start of each 1d candle (00:00 UTC)
        if i > 0 and prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
            # Previous 1d candle
            prev_high = high_1d[i-1] if not np.isnan(high_1d[i-1]) else np.nan
            prev_low = low_1d[i-1] if not np.isnan(low_1d[i-1]) else np.nan
            prev_close = close_1d[i-1] if not np.isnan(close_1d[i-1]) else np.nan
            
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                # Camarilla levels
                camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 6
                camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 6
                camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 4
                camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 4
            else:
                camarilla_h3 = camarilla_l3 = camarilla_h4 = camarilla_l4 = np.nan
        else:
            # Carry forward previous day's levels
            if i == 1:
                camarilla_h3 = camarilla_l3 = camarilla_h4 = camarilla_l4 = np.nan
        
        # Skip if pivot levels not available
        if np.isnan(camarilla_h3) or np.isnan(camarilla_l3):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout signals
        camarilla_up = close_12h[i] > camarilla_h3
        camarilla_down = close_12h[i] < camarilla_l3
        
        # Exit conditions: Opposite Camarilla breakout (L4 for long, H4 for short) or chop regime shifts to ranging (CHOP > 60)
        exit_long = camarilla_down or (chop_aligned[i] > 60.0)
        exit_short = camarilla_up or (chop_aligned[i] > 60.0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Camarilla breakout up + volume confirmation + trending regime
            if camarilla_up and volume_confirm and regime_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Camarilla breakout down + volume confirmation + trending regime
            elif camarilla_down and volume_confirm and regime_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Camarilla breakout OR chop regime shifts to ranging
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals