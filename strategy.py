#!/usr/bin/env python3
"""
6h_ADX_DMI_VolumeRegime_v1
Hypothesis: 6h ADX trend strength with DMI crossover and volume confirmation on 1d regime.
- Long when ADX > 25 (strong trend) AND +DI crosses above -DI AND price above 1d VWAP AND volume spike
- Short when ADX > 25 AND -DI crosses above +DI AND price below 1d VWAP AND volume spike
- Uses 1d VWAP as dynamic support/resistance to filter counter-trend signals
- Volume spike (2.0x 20-period average) confirms institutional participation
- Targets 50-150 trades over 4 years (12-37/year) with discrete position sizing
- Designed to work in both bull (trend following) and bear (strong downtrends) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for VWAP regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14-period) and DI components
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan)
        if len(values) >= period:
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di_smoothed = wilders_smoothing(plus_dm, period)
    minus_di_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI values
    plus_di = 100 * plus_di_smoothed / atr
    minus_di = 100 * minus_di_smoothed / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # DI crossover signals
    plus_di_prev = np.roll(plus_di, 1)
    minus_di_prev = np.roll(minus_di, 1)
    plus_di_prev[0] = np.nan
    minus_di_prev[0] = np.nan
    
    di_cross_up = (plus_di > minus_di) & (plus_di_prev <= minus_di_prev)
    di_cross_down = (minus_di > plus_di) & (minus_di_prev <= plus_di_prev)
    
    # Calculate 1d VWAP for regime filter
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_num = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_den = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_num / vwap_den
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate volume spike (20-period volume average on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 2*period for ADX/DI stability)
    start_idx = 2 * period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(di_cross_up[i]) or np.isnan(di_cross_down[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ADX trend strength with DMI crossover and volume confirmation
        if position == 0:
            # Long: ADX > 25 (strong trend) AND +DI crosses above -DI AND price above 1d VWAP AND volume spike
            if adx[i] > 25 and di_cross_up[i] and close[i] > vwap_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 AND -DI crosses above +DI AND price below 1d VWAP AND volume spike
            elif adx[i] > 25 and di_cross_down[i] and close[i] < vwap_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX falls below 20 (trend weakening) OR -DI crosses above +DI OR price falls below 1d VWAP
            if adx[i] < 20 or di_cross_down[i] or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX falls below 20 OR +DI crosses above -DI OR price rises above 1d VWAP
            if adx[i] < 20 or di_cross_up[i] or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_VolumeRegime_v1"
timeframe = "6h"
leverage = 1.0