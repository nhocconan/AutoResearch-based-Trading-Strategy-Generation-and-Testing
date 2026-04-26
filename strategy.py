#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeRegimeFilter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d trend and volume regime filter.
In bull markets (1d EMA50 up): look for longs when Bull Power > 0 and rising, with volume > 1.3x average.
In bear markets (1d EMA50 down): look for shorts when Bear Power > 0 and rising, with volume > 1.3x average.
Volume regime filter avoids low-conviction breakouts. Discrete sizing (0.0, ±0.25) minimizes fee churn.
Targets 12-25 trades/year (~50-100 over 4 years) to avoid fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(13) and EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(50), 6h EMA(13), volume MA(20)
    start_idx = max(50, 13, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 1.3  # volume at least 1.3x average
        trend_1d_up = ema_50_1d_aligned[i] > ema_13_1d_aligned[i]  # 1d EMA50 > EMA13 = uptrend
        trend_1d_down = ema_50_1d_aligned[i] < ema_13_1d_aligned[i]  # 1d EMA50 < EMA13 = downtrend
        
        # Elder Ray momentum: rising power indicates strengthening move
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_rising = bear_power[i] > bear_power[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 AND rising AND 1d uptrend AND volume confirmation
            long_signal = (bull_power[i] > 0) and bull_power_rising and trend_1d_up and vol_confirmed
            
            # Short: Bear Power > 0 AND rising AND 1d downtrend AND volume confirmation
            short_signal = (bear_power[i] > 0) and bear_power_rising and trend_1d_down and vol_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR 1d trend flips down OR price hits ATR stoploss
            if (bull_power[i] <= 0) or (not trend_1d_up) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power <= 0 OR 1d trend flips up OR price hits ATR stoploss
            if (bear_power[i] <= 0) or (not trend_1d_down) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeRegimeFilter"
timeframe = "6h"
leverage = 1.0