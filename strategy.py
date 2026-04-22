#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and 1d EMA50 trend filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA as in classic Elder Ray).
# Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and price > 1d EMA50.
# Short when Bear Power > 0, Bull Power < 0, ADX > 25, and price < 1d EMA50.
# Exit when ADX < 20 (range) or power signals weaken.
# Uses 1d data for ADX and EMA50 to avoid whipsaw in lower timeframes.
# Target: 20-35 trades/year to stay within fee limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX, EMA50, and Elder Ray components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (using close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power_1d = high_1d - ema13_1d  # High - EMA13
    bear_power_1d = ema13_1d - low_1d   # EMA13 - Low
    
    # Calculate ADX (14-period)
    # +DM, -DM, TR
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Pad arrays to match original length
    plus_dm_padded = np.concatenate([[np.nan], plus_dm])
    minus_dm_padded = np.concatenate([[np.nan], minus_dm])
    tr_padded = np.concatenate([[np.nan], tr])
    
    atr = wilder_smooth(tr_padded, 14)
    plus_di = 100 * wilder_smooth(plus_dm_padded, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm_padded, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        ema50 = ema50_aligned[i]
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, ADX > 25, price > EMA50
            if bull_power > 0 and bear_power < 0 and adx_val > 25 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0, Bull Power < 0, ADX > 25, price < EMA50
            elif bear_power > 0 and bull_power < 0 and adx_val > 25 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: ADX < 20 (range) or power signals weaken
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when ADX < 20 or Bull Power <= 0
                if adx_val < 20 or bull_power <= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when ADX < 20 or Bear Power <= 0
                if adx_val < 20 or bear_power <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_EMA50"
timeframe = "6h"
leverage = 1.0