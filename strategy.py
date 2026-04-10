#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long: Price breaks above Camarilla H3 level + 1d volume > 1.5x 20-period average + chop > 61.8 (range regime)
# - Short: Price breaks below Camarilla L3 level + same volume and chop conditions
# - Exit: Close-based reversal - exit when price crosses Camarilla H4/L4 levels
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Uses 1d Camarilla pivots calculated from previous 1d OHLC
# - Choppiness index (14) > 61.8 indicates ranging market suitable for mean reversion at pivot levels
# - Volume confirmation ensures breakout has conviction
# - Target: 100-180 total trades over 4 years (25-45/year) to stay within HARD MAX: 400 total
# - Works in both bull and bear: chop filter avoids strong trends, pivot levels provide mean reversion edges

name = "4h_1d_camarilla_pivot_volume_chop_v1"
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
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low)
    # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    hl_range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * hl_range_1d
    camarilla_h3 = close_1d + 1.125 * hl_range_1d
    camarilla_l3 = close_1d - 1.125 * hl_range_1d
    camarilla_l4 = close_1d - 1.5 * hl_range_1d
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d current volume for confirmation
    volume_1d_current_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate 4h ATR (14-period) for stoploss
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
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # where ATR1 = True Range
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = 0
    
    atr_14_1d = wilders_smoothing(tr_1d, 14)
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.zeros_like(atr_14_1d)
    for i in range(13, len(atr_14_1d)):
        sum_atr_14[i] = np.sum(atr_14_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (14 * log10(14))) / log10(14)
    chop_1d = np.zeros_like(atr_14_1d)
    for i in range(13, len(atr_14_1d)):
        if sum_atr_14[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_atr_14[i] / (14 * np.log10(14))) / np.log10(14)
        else:
            chop_1d[i] = 0
    
    # Align chop to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_current_aligned[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period average
        volume_confirmation = volume_1d_current_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # Choppiness regime filter: chop > 61.8 indicates ranging market
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + chop filter
            if (close_price > camarilla_h3_aligned[i] and volume_confirmation and chop_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + chop filter
            elif (close_price < camarilla_l3_aligned[i] and volume_confirmation and chop_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_4h[i]
                # Exit conditions: price < Camarilla L4 OR stoploss hit
                if close_price < camarilla_l4_aligned[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Camarilla H4 OR stoploss hit
                if close_price > camarilla_h4_aligned[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals