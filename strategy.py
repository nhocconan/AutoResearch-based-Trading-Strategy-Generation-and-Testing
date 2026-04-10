#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long: price breaks above Camarilla H3 (1d) + 1d volume > 1.5x 20-period MA + chop > 61.8 (range)
# - Short: price breaks below Camarilla L3 (1d) + 1d volume > 1.5x 20-period MA + chop > 61.8 (range)
# - Exit: close-based reversal - exit long when price < Camarilla L3, exit short when price > Camarilla H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 12h
# - Position sizing: 0.25 (discrete level)
# - Uses Camarilla pivots from 1d for structure, 1d volume for confirmation, chop filter to avoid strong trends
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within HARD MAX: 200 total
# - Works in both bull and bear: chop filter identifies ranging markets where Camarilla levels hold, volume confirms genuine breakouts

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR (14-period) for 12h stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
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
    
    atr_14_12h = wilders_smoothing(tr, 14)
    
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
    
    # Calculate Camarilla pivot levels from 1d data
    # Camarilla: based on previous day's high, low, close
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use H3 and L3 for breakouts
    
    # Shift to use previous day's data for today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First value has no previous day
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_h3 = prev_close_1d + 1.0 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.0 * (prev_high_1d - prev_low_1d)
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for calculations)
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get aligned 1d data for current 12h bar (completed 1d bar)
        volume_1d_current = volume_1d[i // 2] if i // 2 < len(volume_1d) else volume_1d[-1]  # 2x 12h bars = 1d bar
        volume_ma_current = volume_ma_aligned[i]
        chop_current = chop_aligned[i]
        camarilla_h3_current = camarilla_h3_aligned[i]
        camarilla_l3_current = camarilla_l3_aligned[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period MA of 1d volume
        # Since we're on 12h timeframe, we compare 12h volume to 1d volume MA (approximation)
        volume_spike = volume_12h[i] > 1.5 * volume_ma_current
        
        # Chop regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop_current > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume spike + chop filter
            if (close_price > camarilla_h3_current and volume_spike and chop_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume spike + chop filter
            elif (close_price < camarilla_l3_current and volume_spike and chop_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_12h[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3_current or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_12h[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3_current or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals