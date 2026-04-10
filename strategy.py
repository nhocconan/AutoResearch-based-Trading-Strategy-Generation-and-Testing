#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ATR volatility filter and chop regime filter
# - Long: price breaks above Donchian upper (20-period) + 12h ATR(14) > 1.5x 20-period MA + chop > 61.8 (range)
# - Short: price breaks below Donchian lower (20-period) + 12h ATR(14) > 1.5x 20-period MA + chop > 61.8 (range)
# - Exit: close-based reversal - exit long when price < Donchian lower, exit short when price > Donchian upper
# - Stoploss: ATR-based - exit when price moves against position by 2.0 * ATR(14) on 4h
# - Position sizing: 0.25 (discrete level)
# - Uses Donchian channels for structure, 12h ATR for volatility confirmation, chop filter to avoid strong trends
# - Target: 75-150 total trades over 4 years (19-38/year) to stay within HARD MAX: 400 total
# - Works in both bull and bear: chop filter identifies ranging markets where Donchian levels hold, volatility confirms genuine breakouts

name = "4h_12h_donchian_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
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
    
    # Calculate 12h ATR (14-period) for volatility filter
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = 0  # First TR is 0 (no previous close)
    atr_14_12h = wilders_smoothing(tr_12h, 14)
    
    # Calculate 12h ATR moving average (20-period)
    atr_12h_series = pd.Series(atr_14_12h)
    atr_ma_20_12h = atr_12h_series.rolling(window=20, min_periods=20).mean().values
    atr_ma_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_20_12h)
    
    # Calculate True Range for chop
    tr1_12h_chop = high_12h - low_12h
    tr2_12h_chop = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h_chop = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h_chop = np.maximum(tr1_12h_chop, np.maximum(tr2_12h_chop, tr3_12h_chop))
    tr_12h_chop[0] = 0  # First TR is 0 (no previous close)
    
    # Calculate Sum of True Range over 14 periods for chop numerator
    sum_tr_14 = pd.Series(tr_12h_chop).rolling(window=14, min_periods=14).sum().values
    
    # Calculate Max High - Min Low over 14 periods for chop denominator
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Calculate Choppiness Index (CHOP)
    chop = np.where(
        (range_14 != 0) & (sum_tr_14 != 0) & ~np.isnan(range_14) & ~np.isnan(sum_tr_14),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50  # Default to middle when undefined
    )
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate Donchian channels (20-period) from 4h data
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for calculations)
        # Skip if any required data is invalid
        if (np.isnan(atr_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get aligned 12h data for current 4h bar (completed 12h bar)
        atr_12h_current = align_htf_to_ltf(prices, df_12h, atr_14_12h)[i]
        chop_current = chop_aligned[i]
        
        # Volatility spike condition: current 12h ATR > 1.5x 20-period MA
        volatility_spike = atr_12h_current > 1.5 * atr_ma_aligned[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (mean reversion favorable)
        chop_filter = chop_current > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volatility spike + chop filter
            if (close_price > donchian_upper[i] and volatility_spike and chop_filter):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower + volatility spike + chop filter
            elif (close_price < donchian_lower[i] and volatility_spike and chop_filter):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.0 * atr_14_4h[i]
                # Exit conditions: price < Donchian lower OR stoploss hit
                if close_price < donchian_lower[i] or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.0 * atr_14_4h[i]
                # Exit conditions: price > Donchian upper OR stoploss hit
                if close_price > donchian_upper[i] or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals