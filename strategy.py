#!/usr/bin/env python3
"""
Experiment #4971: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, Williams %R(14) extreme readings (<20 for oversold, >80 for overbought) 
combined with 1d ADX(14) > 25 for strong trend confirmation and volume >1.5x average captures 
high-probability mean-reversion entries within strong trends. Works in bull markets (buy dips in uptrend) 
and bear markets (sell rallies in downtrend). Designed for 12-37 trades/year on 6h timeframe 
(50-150 total over 4 years) to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4971_6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for trend strength filter ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(values, period):
            smoothed = np.full_like(values, np.nan)
            if len(values) >= period:
                # First value is simple average
                smoothed[period-1] = np.nanmean(values[:period])
                # Subsequent values: smoothed_prev * (1 - 1/period) + current * (1/period)
                alpha = 1.0 / period
                for i in range(period, len(values)):
                    if not np.isnan(smoothed[i-1]):
                        smoothed[i] = smoothed[i-1] * (1 - alpha) + values[i] * alpha
                    else:
                        smoothed[i] = np.nan
            return smoothed
        
        atr_1d = wilders_smoothing(tr_1d, 14)
        plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
        minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilders_smoothing(dx_1d, 14)
        
        # Handle division by zero or near-zero
        adx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, adx_1d)
    else:
        adx_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF ADX to 6h timeframe
    if len(adx_1d) > 0:
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(14, 20)  # Williams %R, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse on opposite signal or ADX weakening ---
        if in_position:
            # Exit conditions: reverse signal or ADX < 20 (trend weakening)
            long_exit = (williams_r[i] > -20) or (adx_1d_aligned[i] < 20)
            short_exit = (williams_r[i] < -80) or (adx_1d_aligned[i] < 20)
            
            if (position_side > 0 and long_exit) or (position_side < 0 and short_exit):
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = SIZE * position_side  # Maintain position
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Williams %R extreme readings for mean reversion within trend
        oversold = williams_r[i] < -80  # Extremely oversold
        overbought = williams_r[i] > -20  # Extremely overbought
        
        # Entry conditions: mean reversion in direction of trend
        # In uptrend (ADX strong), buy oversold dips
        # In downtrend (ADX strong), sell overbought rallies
        if strong_trend and vol_confirm:
            if oversold:
                # Buy the dip in strong uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif overbought:
                # Sell the rally in strong uptrend (or buy dip in downtrend context)
                # Actually, in strong trend, we want to continue with trend, so:
                # For simplicity, we'll mean-revert against extreme readings
                # But let's adjust: in strong trend, we fade extremes only if counter-trend
                # Better approach: use Williams %R for entry timing in trend direction
                # Re-thinking: Williams %R > -20 in uptrend suggests overbought -> wait for pullback
                # Williams %R < -80 in downtrend suggests oversold -> wait for bounce
                # So we actually want:
                # Uptrend: buy when Williams %R crosses back above -80 from below
                # Downtrend: sell when Williams %R crosses back below -20 from above
                # But for simplicity and to avoid lookahead, we'll use:
                # Enter long when oversold AND we detect potential uptrend (price > previous close)
                # Enter short when overbought AND we detect potential downtrend (price < previous close)
                if price > close[i-1]:  # Recent upward momentum
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
                elif price < close[i-1]:  # Recent downward momentum
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals