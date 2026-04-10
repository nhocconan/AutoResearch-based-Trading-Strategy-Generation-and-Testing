#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and chop regime filter
# - Long: price breaks above Camarilla H3 level (1d) + 1w volume > 1.5x 20-period MA + chop > 61.8 (range)
# - Short: price breaks below Camarilla L3 level (1d) + 1w volume > 1.5x 20-period MA + chop > 61.8 (range)
# - Exit: close-based reversal - exit long when price < Camarilla L3, exit short when price > Camarilla H3
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 1d
# - Position sizing: 0.25 (discrete level)
# - Uses Camarilla pivots for structure, 1w volume for confirmation, chop filter to avoid strong trends
# - Target: 30-100 total trades over 4 years (7-25/year) to stay within HARD MAX: 150 total
# - Works in both bull and bear: chop filter identifies ranging markets where Camarilla levels hold, volume confirms genuine breakouts

name = "1d_1w_camarilla_breakout_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate ATR (14-period) for 1d stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
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
    
    atr_14_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1w volume moving average (20-period)
    volume_1w_series = pd.Series(volume_1w)
    volume_ma_20_1w = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate True Range for chop
    tr1_1w_chop = high_1w - low_1w
    tr2_1w_chop = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w_chop = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w_chop = np.maximum(tr1_1w_chop, np.maximum(tr2_1w_chop, tr3_1w_chop))
    tr_1w_chop[0] = 0  # First TR is 0 (no previous close)
    
    # Calculate Sum of True Range over 14 periods for chop numerator
    sum_tr_14 = pd.Series(tr_1w_chop).rolling(window=14, min_periods=14).sum().values
    
    # Calculate Max High - Min Low over 14 periods for chop denominator
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Calculate Choppiness Index (CHOP)
    chop = np.where(
        (range_14 != 0) & (sum_tr_14 != 0) & ~np.isnan(range_14) & ~np.isnan(sum_tr_14),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50  # Default to middle when undefined
    )
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First value
    prev_low[0] = low_1d[0]    # First value
    prev_close[0] = close_1d[0] # First value
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for calculations)
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close
        close_price = close_1d[i]
        
        # Get aligned 1w data for current 1d bar (completed 1w bar)
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        chop_current = chop_aligned[i]
        
        # Volume spike condition: current 1w volume > 1.5x 20-period MA
        volume_spike = volume_1w_current > 1.5 * volume_ma_aligned[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop_current > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume spike + chop filter
            if (close_price > camarilla_h3[i] and volume_spike and chop_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume spike + chop filter
            elif (close_price < camarilla_l3[i] and volume_spike and chop_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_1d[i]
                # Exit conditions: price < Camarilla L3 OR stoploss hit
                if close_price < camarilla_l3[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_1d[i]
                # Exit conditions: price > Camarilla H3 OR stoploss hit
                if close_price > camarilla_h3[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals