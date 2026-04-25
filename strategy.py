#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_RegimeFilter
Hypothesis: 12-hour Camarilla R3/S3 level breakout with 1-week EMA50 trend filter and choppiness regime filter.
Long when price breaks above R3 with volume confirmation in 1-week uptrend and low chop regime.
Short when price breaks below S3 with volume confirmation in 1-week downtrend and low chop regime.
Exit via ATR trailing stop (2.5*ATR from extreme) or opposite Camarilla level (S3 for longs, R3 for shorts).
Designed for ~80-120 trades over 4 years (20-30/year) via tight Camarilla breakout conditions.
1-week trend filter ensures alignment with higher timeframe bias. Choppiness filter avoids whipsaws in ranging markets.
Volume confirmation ensures breakouts have conviction. Works in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need 50 for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    # Choppiness regime filter (14-period)
    chop_period = 14
    true_range = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = highest_high - lowest_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(true_range / chop_denom) / np.log10(chop_period)
    chop_regime = chop < 61.8  # low chop = trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, chop_period, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        vol_ok = vol_regime[i]
        chop_ok = chop_regime[i]
        
        # Calculate Camarilla levels from previous 1d bar
        # Need to get the previous completed 1d bar's OHLC
        # We'll use the 1d data aligned to current time, but shift by 1 to avoid look-ahead
        if i < len(prices):
            # Get the 1d index for current time (already aligned via align_htf_to_ltf logic)
            # We need to access the 1d OHLC values properly
            pass  # Will handle in the actual calculation below
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter + low chop)
            if ema_trend > 0 and chop_ok:  # 1w uptrend regime
                # Calculate Camarilla levels for current 1d bar
                # We need the previous 1d completed bar's OHLC
                # Simplified: use rolling window on 1d aligned data
                # For now, use a proxy: calculate from current 1d aligned high/low/close
                # This is acceptable as we're using completed 1d bars via alignment
                pass  # Will implement properly below
            else:  # 1w downtrend regime
                pass  # Will implement properly below
            
            # Calculate Camarilla levels properly
            # We need to access 1d data for the previous completed day
            # Since we're using 12h timeframe, we can use the 1d aligned data with proper indexing
            # Get the 1d aligned close, high, low series
            # We'll calculate these outside the loop for efficiency
            pass  # Will implement properly below
        
        # For now, implement a simplified version that will be replaced
        if position == 0:
            if ema_trend > close[i] and chop_ok and vol_ok:  # 1w uptrend
                # Simple breakout above recent high as proxy for R3 breakout
                recent_high = pd.Series(high).rolling(window=20, min_periods=20).max().iloc[i] if hasattr(pd.Series(high).rolling(window=20, min_periods=20).max(), 'iloc') else np.nan
                if not np.isnan(recent_high) and close[i] > recent_high:
                    signals[i] = 0.25
                    position = 1
                    long_extreme = close[i]
                else:
                    signals[i] = 0.0
            elif ema_trend < close[i] and chop_ok and vol_ok:  # 1w downtrend
                # Simple breakout below recent low as proxy for S3 breakout
                recent_low = pd.Series(low).rolling(window=20, min_periods=20).min().iloc[i] if hasattr(pd.Series(low).rolling(window=20, min_periods=20).min(), 'iloc') else np.nan
                if not np.isnan(recent_low) and close[i] < recent_low:
                    signals[i] = -0.25
                    position = -1
                    short_extreme = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit: ATR trailing stop or break below recent low
            atr_stop = long_extreme - 2.5 * atr[i]
            recent_low = pd.Series(low).rolling(window=20, min_periods=20).min().iloc[i] if hasattr(pd.Series(low).rolling(window=20, min_periods=20).min(), 'iloc') else np.nan
            if not np.isnan(recent_low) and (close[i] <= atr_stop or close[i] < recent_low):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            signals[i] = -0.25
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit: ATR trailing stop or break above recent high
            atr_stop = short_extreme + 2.5 * atr[i]
            recent_high = pd.Series(high).rolling(window=20, min_periods=20).max().iloc[i] if hasattr(pd.Series(high).rolling(window=20, min_periods=20).max(), 'iloc') else np.nan
            if not np.isnan(recent_high) and (close[i] >= atr_stop or close[i] > recent_high):
                signals[i] = 0.0
                position = 0
    
    return signals

# Proper implementation with correct Camarilla calculation
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need 50 for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    # Choppiness regime filter (14-period)
    chop_period = 14
    tr_series = pd.Series(tr)
    true_range = tr_series.rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = highest_high - lowest_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(true_range / chop_denom) / np.log10(chop_period)
    chop_regime = chop < 61.8  # low chop = trending regime
    
    # Pre-calculate 1d aligned series for Camarilla calculation
    # We need the previous completed 1d bar's OHLC for each 12h bar
    # Since 1d -> 12h: 2x multiplier, we can shift the 1d aligned data by 2 bars
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Shift by 2 bars to get previous completed 1d bar (avoid look-ahead)
    # For 12h timeframe, 1d bar = 2 bars, so previous completed 1d bar = current index - 2
    close_1d_prev = np.roll(close_1d_aligned, 2)
    high_1d_prev = np.roll(high_1d_aligned, 2)
    low_1d_prev = np.roll(low_1d_aligned, 2)
    # First 2 bars will have invalid data (rolled from end), but we'll check min_periods later
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, chop_period, 50, 2)  # +2 for the shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            np.isnan(close_1d_prev[i]) or np.isnan(high_1d_prev[i]) or np.isnan(low_1d_prev[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        vol_ok = vol_regime[i]
        chop_ok = chop_regime[i]
        
        # Calculate Camarilla levels from previous completed 1d bar
        # Camarilla formula:
        # R4 = close + (high - low) * 1.1/2
        # R3 = close + (high - low) * 1.1/4
        # R2 = close + (high - low) * 1.1/6
        # R1 = close + (high - low) * 1.1/12
        # PP = (high + low + close) / 3
        # S1 = close - (high - low) * 1.1/12
        # S2 = close - (high - low) * 1.1/6
        # S3 = close - (high - low) * 1.1/4
        # S4 = close - (high - low) * 1.1/2
        
        prev_high = high_1d_prev[i]
        prev_low = low_1d_prev[i]
        prev_close = close_1d_prev[i]
        
        range_hl = prev_high - prev_low
        if range_hl <= 0:
            # Invalid range, skip signal generation
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        r3 = prev_close + range_hl * 1.1 / 4
        s3 = prev_close - range_hl * 1.1 / 4
        
        if position == 0:
            # Only trade in trending regimes (1w EMA50 filter + low chop)
            if ema_trend > prev_close and chop_ok and vol_ok:  # 1w uptrend regime
                # Long: break above R3 with volume confirmation
                if close[i] > r3:
                    signals[i] = 0.25
                    position = 1
                    long_extreme = close[i]
                else:
                    signals[i] = 0.0
            elif ema_trend < prev_close and chop_ok and vol_ok:  # 1w downtrend regime
                # Short: break below S3 with volume confirmation
                if close[i] < s3:
                    signals[i] = -0.25
                    position = -1
                    short_extreme = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below S3 (opposite Camarilla level)
            if close[i] <= atr_stop or close[i] < s3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above R3 (opposite Camarilla level)
            if close[i] >= atr_stop or close[i] > r3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_RegimeFilter"
timeframe = "12h"
leverage = 1.0