#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + chop regime filter
    # Long when: price breaks above Camarilla H3 (1d) AND 1d volume > 2x 20-day avg volume AND CHOP(14) < 38.2 (trending)
    # Short when: price breaks below Camarilla L3 (1d) AND 1d volume > 2x 20-day avg volume AND CHOP(14) < 38.2
    # Exit when: price crosses Camarilla Pivot point (1d) OR CHOP(14) > 61.8 (range)
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Camarilla levels provide institutional support/resistance; volume confirms breakout strength; chop filter avoids false signals in ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # H4 = C + 1.5*(H-L), H3 = C + 1.25*(H-L), H2 = C + 1.0*(H-L), H1 = C + 0.75*(H-L)
    # L1 = C - 0.75*(H-L), L2 = C - 1.0*(H-L), L3 = C - 1.25*(H-L), L4 = C - 1.5*(H-L)
    # Pivot = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's data to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first day's values to NaN (no previous day)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_h3 = prev_close_1d + 1.25 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.25 * (prev_high_1d - prev_low_1d)
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d volume confirmation: volume > 2x 20-day average volume
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (2.0 * avg_vol_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate Choppiness Index on 4h data (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / log10( (max(high,14) - min(low,14)) ) )
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_values = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop_denominator = np.log10(max_high - min_low)
    chop_numerator = np.log10(np.sum(atr_values.reshape(-1, atr_period), axis=1).flatten())
    # Handle edge cases for chop calculation
    chop = np.zeros_like(close)
    chop.fill(np.nan)
    valid_idx = (chop_denominator > 0) & ~np.isnan(chop_denominator) & ~np.isnan(chop_numerator)
    chop[valid_idx] = 100 - (100 * chop_numerator[valid_idx] / chop_denominator[valid_idx])
    
    # Regime filter: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid breakouts)
    trending_regime = chop < 38.2
    ranging_regime = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # break above previous H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # break below previous L3
        
        # Entry conditions with volume confirmation and trending regime
        long_entry = breakout_up and volume_spike_aligned[i] and trending_regime[i] and position != 1
        short_entry = breakout_down and volume_spike_aligned[i] and trending_regime[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < camarilla_pivot_aligned[i] or ranging_regime[i]))
        exit_short = (position == -1 and (close[i] > camarilla_pivot_aligned[i] or ranging_regime[i]))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0