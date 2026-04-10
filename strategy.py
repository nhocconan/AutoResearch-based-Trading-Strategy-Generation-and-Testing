#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long: price breaks above Camarilla H3 level + 1d volume > 1.5x 20-period average + chop > 61.8 (range)
# - Short: price breaks below Camarilla L3 level + 1d volume > 1.5x 20-period average + chop > 61.8 (range)
# - Exit: close-based reversal - exit long when price < Camarilla H3, exit short when price > Camarilla L3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Uses Camarilla pivots for structure, volume confirmation for conviction, chop filter to avoid trending markets
# - Target: 75-150 total trades over 4 years (19-38/year) to stay within HARD MAX: 400 total
# - Works in both bull and bear: chop filter identifies ranging markets where pivot levels hold, volume confirms genuine breakouts

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR (14-period) for 4h stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is 0 (no previous close)
    
    # Wilder's ATR smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])  # First value is simple average
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_14_4h = wilders_smoothing(tr, 14)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate True Range for chop
    tr1_1d_chop = high_1d - low_1d
    tr2_1d_chop = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d_chop = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d_chop = np.maximum(tr1_1d_chop, np.maximum(tr2_1d_chop, tr3_1d_chop))
    tr_1d_chop[0] = 0  # First TR is 0 (no previous close)
    
    # Calculate Sum of True Range over 14 periods for chop numerator
    sum_tr_14 = pd.Series(tr_1d_chop).rolling(window=14, min_periods=14).sum().values
    
    # Calculate Max High - Min Low over 14 periods for chop denominator
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Calculate Choppiness Index (CHOP)
    chop = np.where(
        (range_14 != 0) & (sum_tr_14 != 0) & ~np.isnan(range_14) & ~np.isnan(sum_tr_14),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50  # Default to middle when undefined
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for volume MA)
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d data for Camarilla calculation (use previous completed 1d bar)
        # Get index of completed 1d bar for current 4h bar
        # We need to use the 1d bar that closed before or at the current 4h bar
        # Since we're using align_htf_to_ltf, we can get the aligned values
        
        # Calculate Camarilla levels from previous 1d bar
        # We'll use the 1d data shifted by 1 to avoid look-ahead
        if i >= 16:  # Need at least one 1d bar (16x 4h bars) to calculate
            # Get the 1d index for current position
            # Since we don't have direct mapping, we'll use the aligned arrays approach
            # For simplicity, we'll use the previous 1d bar's data
            
            # Get aligned 1d OHLC (these will be the values from the completed 1d bar)
            high_1d_prev = align_htf_to_ltf(prices, df_1d, high_1d)[i]
            low_1d_prev = align_htf_to_ltf(prices, df_1d, low_1d)[i]
            close_1d_prev = align_htf_to_ltf(prices, df_1d, close_1d)[i]
            
            # Calculate Camarilla levels
            range_1d = high_1d_prev - low_1d_prev
            camarilla_h3 = close_1d_prev + range_1d * 1.1 / 4
            camarilla_l3 = close_1d_prev - range_1d * 1.1 / 4
            
            # Get current 1d volume for confirmation
            volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
            volume_spike = volume_1d_current > 1.5 * volume_ma_aligned[i]
            
            # Chop regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
            chop_filter = chop_aligned[i] > 61.8
            
            if position == 0:  # Flat - look for new entries
                # Long entry: price breaks above Camarilla H3 + volume spike + chop filter
                if (close_price > camarilla_h3 and volume_spike and chop_filter):
                    position = 1
                    entry_price = close_price
                    signals[i] = 0.25
                # Short entry: price breaks below Camarilla L3 + volume spike + chop filter
                elif (close_price < camarilla_l3 and volume_spike and chop_filter):
                    position = -1
                    entry_price = close_price
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:  # Have position - look for exit or stoploss
                # Calculate stoploss level
                if position == 1:  # Long position
                    stop_loss = entry_price - 2.0 * atr_14_4h[i]
                    # Exit conditions: price < Camarilla H3 OR stoploss hit
                    if close_price < camarilla_h3 or close_price <= stop_loss:
                        position = 0
                        entry_price = 0.0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.25
                else:  # position == -1, Short position
                    stop_loss = entry_price + 2.0 * atr_14_4h[i]
                    # Exit conditions: price > Camarilla L3 OR stoploss hit
                    if close_price > camarilla_l3 or close_price >= stop_loss:
                        position = 0
                        entry_price = 0.0
                        signals[i] = 0.0
                    else:
                        signals[i] = -0.25
        else:
            # Not enough 1d data yet
            signals[i] = 0.0
    
    return signals