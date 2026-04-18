#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with daily volume and volatility filters.
# Camarilla pivot levels provide statistically significant support/resistance based on prior day's range.
# Breakouts above R1 or below S1 indicate strong momentum with follow-through potential.
# Daily volume filter ensures institutional participation, while volatility filter avoids choppy markets.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above R1) and bear markets (breakouts below S1).
name = "12h_Camarilla_R1S1_Breakout_Volume_Volatility"
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
    
    # Get daily data for Camarilla levels and filters (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_d, 1)
    prev_low = np.roll(low_d, 1)
    prev_close = np.roll(close_d, 1)
    
    # Avoid look-ahead: use previous day's data only
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_d = prev_high - prev_low
    r1 = prev_close + range_d * 1.1 / 12
    s1 = prev_close - range_d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Daily ATR (14-period) for volatility filter - avoids trading in low volatility/chop
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold = atr * atr_mult
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Session filter: 08-20 UTC (avoid low volume Asian session)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_threshold_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume above average
        vol_confirm = df_1d['volume'].values[i] > vol_ma_aligned[i]
        
        # Volatility filter: sufficient volatility to avoid chop
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR volatility drops (chop/ranging market)
            exit_condition = close[i] < s1_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR volatility drops (chop/ranging market)
            exit_condition = close[i] > r1_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals