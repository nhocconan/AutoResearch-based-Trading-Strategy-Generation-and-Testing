#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising (2-bar momentum) AND 1d ADX < 25 (range regime).
# Short when Bear Power < 0 and falling (2-bar momentum) AND 1d ADX < 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Works in bull/bear: ADX < 25 filters out strong trends where Elder Ray fails, focuses on range/mean-reversion.

name = "6h_ElderRay_1dADX_Range_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for 08-20 UTC filter (optional, can be removed if too restrictive)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Higher = stronger bulls
    bear_power = low - ema13   # Lower (more negative) = stronger bears
    
    # Elder Ray momentum (2-bar change)
    bull_power_mom = bull_power - np.roll(bull_power, 2)
    bear_power_mom = bear_power - np.roll(bear_power, 2)
    # Handle first two bars
    bull_power_mom[:2] = 0
    bear_power_mom[:2] = 0
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter: <25 = range/weak trend (good for Elder Ray mean reversion)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # First bar
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_plus_1d = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_minus_1d = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_1d / (atr_1d + 1e-10)
    di_minus_1d = 100 * dm_minus_1d / (atr_1d + 1e-10)
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 20  # warmup for Elder Ray and ATR
    
    for i in range(start_idx, n):
        # Optional session filter: 08-20 UTC (uncomment if needed)
        # if not (8 <= hours[i] <= 20):
        #     signals[i] = 0.0
        #     continue
        
        if (np.isnan(atr[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_mom[i]) or np.isnan(bear_power_mom[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Regime filter: only trade in range/weak trend market (ADX < 25)
        regime_filter = adx_1d_aligned[i] < 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND rising momentum AND range regime
            if (bull_power[i] > 0 and 
                bull_power_mom[i] > 0 and 
                regime_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bear Power < 0 AND falling momentum AND range regime
            elif (bear_power[i] < 0 and 
                  bear_power_mom[i] < 0 and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray signals weaken (Bull Power <= 0 or momentum negative) OR regime shifts to strong trend
            elif (bull_power[i] <= 0 or bull_power_mom[i] <= 0) or \
                 adx_1d_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Ray signals weaken (Bear Power >= 0 or momentum positive) OR regime shifts to strong trend
            elif (bear_power[i] >= 0 or bear_power_mom[i] >= 0) or \
                 adx_1d_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals