#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme + 1d Volume Spike + Chop Regime Filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) on 12h
# Entry confirmed by 1d volume > 1.5 * 20-period average volume (spike)
# Regime filter: 1d Chopiness Index(14) > 61.8 = ranging (mean revert), < 38.2 = trending (follow breakout)
# Designed for low frequency (50-150 trades over 4 years) with clear reversal logic in ranging markets
# and breakout logic in trending markets, using volume confirmation to avoid false signals

name = "12h_WilliamsR_1dVolume_Chop_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for regime and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 20 for vol avg + 14 for chop
        return np.zeros(n)
    
    # 1d volume spike confirmation: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.5 * vol_ma20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Chopiness Index(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Sum of True Range over 14 periods
    tr_sum14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(tr_sum14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    hh_ll = hh14 - ll14
    chop_raw = np.where((hh_ll > 0) & (tr_sum14 > 0), 
                        100 * np.log10(tr_sum14 / hh_ll) / np.log10(14), 
                        50)  # default to neutral when undefined
    chop = np.concatenate([[np.nan] * 13, chop_raw[13:]])  # align with 14-period lookback
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h Williams %R(14)
    highest_high12 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low12 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high12 - close) / (highest_high12 - lowest_low12)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high12 - lowest_low12) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: max(34 for 1d indicators, 14 for Williams %R)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters from 1d Chop
        ranging = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            if ranging:
                # Mean reversion in ranging market
                # Long: Williams %R deeply oversold
                if williams_r[i] < -80 and volume_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R deeply overbought
                elif williams_r[i] > -20 and volume_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif trending:
                # Breakout continuation in trending market
                # Long: Williams %R rising from oversold (bullish momentum)
                if williams_r[i] < -50 and williams_r[i] > williams_r[i-1] and volume_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R falling from overbought (bearish momentum)
                elif williams_r[i] > -50 and williams_r[i] < williams_r[i-1] and volume_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (Chop 38.2-61.8) - stay flat to avoid whipsaw
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if ranging:
                # Exit long when Williams %R reaches overbought territory (mean reversion complete)
                if williams_r[i] > -20:
                    exit_long = True
            elif trending:
                # Exit long when Williams %R shows bearish momentum (failure to make new highs)
                if williams_r[i] < williams_r[i-1] and williams_r[i] < -50:
                    exit_long = True
            else:
                # Transition regime - exit on any deterioration
                if williams_r[i] > -50:  # Lost momentum
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if ranging:
                # Exit short when Williams %R reaches oversold territory (mean reversion complete)
                if williams_r[i] < -80:
                    exit_short = True
            elif trending:
                # Exit short when Williams %R shows bullish momentum (failure to make new lows)
                if williams_r[i] > williams_r[i-1] and williams_r[i] > -50:
                    exit_short = True
            else:
                # Transition regime - exit on any deterioration
                if williams_r[i] < -50:  # Lost momentum
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals