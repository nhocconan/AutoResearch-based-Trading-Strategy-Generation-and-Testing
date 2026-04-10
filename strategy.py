#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.8x 20-period 1d volume SMA AND Choppiness Index > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.8x 20-period 1d volume SMA AND Choppiness Index > 61.8 (ranging market)
# - Exit: Price reversion to Camarilla Pivot Point (PP) OR ATR trailing stop (1.5x ATR from extreme)
# - Uses 1d for volume confirmation and volatility regime, 4h for precise Camarilla levels and entry timing
# - Position sizing: 0.25 discrete level to balance profit potential and drawdown control
# - Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Camarilla pivots work well in ranging markets (common in 2025 BTC/ETH bear/range) with volume confirmation filtering false breakouts
# - Choppiness filter ensures we only trade in ranging regimes where mean reversion at pivots is effective

name = "4h_1d_camarilla_breakout_chop_v1"
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
    
    # Calculate 1d Choppiness Index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * LOG10(sumTR14 / (HH14 - LL14)) / LOG10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h ATR for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels are calculated from previous day's price action
    # We need to align the 1d OHLC to 4h bars
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # H4 = C + ((H-L) * 1.1/2)
    # H3 = C + ((H-L) * 1.1/4)
    # H2 = C + ((H-L) * 1.1/6)
    # H1 = C + ((H-L) * 1.1/12)
    # L1 = C - ((H-L) * 1.1/12)
    # L2 = C - ((H-L) * 1.1/6)
    # L3 = C - ((H-L) * 1.1/4)
    # L4 = C - ((H-L) * 1.1/2)
    range_1d = h_1d - l_1d
    camarilla_pp = (h_1d + l_1d + c_1d) / 3.0  # Pivot Point
    camarilla_h3 = camarilla_pp + (range_1d * 1.1 / 4)
    camarilla_l3 = camarilla_pp - (range_1d * 1.1 / 4)
    camarilla_h4 = camarilla_pp + (range_1d * 1.1 / 2)
    camarilla_l4 = camarilla_pp - (range_1d * 1.1 / 2)
    camarilla_h6 = camarilla_pp + (range_1d * 1.1 / 6)  # H2
    camarilla_l6 = camarilla_pp - (range_1d * 1.1 / 6)  # L2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
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
        if (np.isnan(atr[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (need to align properly)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        # Volume confirmation: 1d volume > 1.8x 20-period 1d volume SMA
        vol_confirm = vol_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: ranging market (Choppiness Index > 61.8)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Break above H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Break below L3
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above Camarilla H3 AND volume confirmation AND ranging regime
            if breakout_up and vol_confirm and chop_regime:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below Camarilla L3 AND volume confirmation AND ranging regime
            elif breakout_down and vol_confirm and chop_regime:
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 1.5*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 1.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to Camarilla Pivot Point
            exit_condition = (close[i] < trailing_stop) or (close[i] < camarilla_pp_aligned[i])
            
            if exit_condition:
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
            
            # ATR trailing stop: exit if price rises 1.5*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 1.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to Camarilla Pivot Point
            exit_condition = (close[i] > trailing_stop) or (close[i] > camarilla_pp_aligned[i])
            
            if exit_condition:
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