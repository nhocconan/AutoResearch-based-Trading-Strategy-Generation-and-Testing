#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period 1d volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period 1d volume SMA AND chop > 61.8
# - Exit: price returns to Camarilla Pivot level or opposing Camarilla breakout
# - Uses 1d for volume and chop calculation, 12h for price action and Camarilla levels
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 12-30 trades/year (48-120 total over 4 years) to avoid overtrading
# - Camarilla pivots work well in ranging markets; volume confirmation ensures institutional participation
# - Choppiness filter avoids false breakouts in strong trends

name = "12h_1d_camarilla_volspike_chop_v2"
timeframe = "12h"
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
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d choppiness index (14-period)
    # Chop = 100 * log10(sum(ATR14) / (log10(highest_high - lowest_low) * 14))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    mask = (range_14 > 0) & (~np.isnan(sum_atr14)) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(sum_atr14[mask] / (np.log10(range_14[mask]) * 14))
    chop = np.where(~mask, 50.0, chop)  # default to neutral when invalid
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-calculate 12h Camarilla levels (using previous day's OHLC)
    # We need to align 1d OHLC to 12h bars
    df_1d_ohlc = df_1d[['open', 'high', 'low', 'close']]
    open_1d = df_1d_ohlc['open'].values
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    # Align 1d OHLC to 12h timeframe
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each 12h bar using previous 1d close
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #            L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    #            Pivot = (high + low + close)/3
    rangep = high_1d_aligned - low_1d_aligned
    camarilla_pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    camarilla_h3 = camarilla_pivot + 1.1 * rangep / 4.0
    camarilla_l3 = camarilla_pivot - 1.1 * rangep / 4.0
    camarilla_h4 = camarilla_pivot + 1.1 * rangep / 2.0
    camarilla_l4 = camarilla_pivot - 1.1 * rangep / 2.0
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_since_entry = 0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_pivot[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period 1d volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Choppiness filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for entry
            # Require both volume confirmation and chop filter
            if vol_confirm and chop_filter:
                # Long: price breaks above Camarilla H3
                if close[i] > camarilla_h3[i]:
                    position = 1
                    signals[i] = 0.25
                    bars_since_entry = 0
                # Short: price breaks below Camarilla L3
                elif close[i] < camarilla_l3[i]:
                    position = -1
                    signals[i] = -0.25
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # In position - look for exit
            bars_since_entry += 1
            
            # Exit conditions: price returns to Camarilla Pivot or opposing breakout
            exit_signal = False
            if position == 1:  # Long position
                if close[i] <= camarilla_pivot[i]:  # Return to pivot
                    exit_signal = True
                elif close[i] < camarilla_l3[i]:  # Opposing breakdown
                    exit_signal = True
            elif position == -1:  # Short position
                if close[i] >= camarilla_pivot[i]:  # Return to pivot
                    exit_signal = True
                elif close[i] > camarilla_h3[i]:  # Opposing breakout
                    exit_signal = True
            
            # Optional: time-based exit (max 8 bars = 4 days)
            if bars_since_entry >= 8:
                exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals