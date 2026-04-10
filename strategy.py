#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 and Bear Power < 0 and 1d ADX > 25 (strong trend)
# - Short when Bear Power > 0 and Bull Power < 0 and 1d ADX > 25 (strong trend)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Bull Power and Bear Power
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # 1d ADX(14) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    def ma_wilder(arr, period):
        res = np.zeros_like(arr)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    atr_14_1d = ma_wilder(tr, 14)
    dm_plus_smooth = ma_wilder(dm_plus, 14)
    dm_minus_smooth = ma_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_14_1d + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_14_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_14_1d = ma_wilder(dx, 14)
    
    # Align HTF indicators to LTF
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # LTF ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1_ltf = high - low
    tr2_ltf = np.abs(high - np.roll(close, 1))
    tr3_ltf = np.abs(low - np.roll(close, 1))
    tr_ltf = np.maximum(tr1_ltf, np.maximum(tr2_ltf, tr3_ltf))
    tr_ltf[0] = tr1_ltf[0]
    
    atr_14_ltf = np.zeros_like(tr_ltf)
    atr_14_ltf[14-1] = np.mean(tr_ltf[:14])
    for i in range(14, len(tr_ltf)):
        atr_14_ltf[i] = (atr_14_ltf[i-1] * (14-1) + tr_ltf[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(atr_14_ltf[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - 2.0 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + 2.0 * entry_atr:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with ADX trend filter
            if adx_14_1d_aligned[i] > 25:  # Strong trend
                # Long signal: Bull Power > 0 and Bear Power < 0
                if bull_power_1d_aligned[i] > 0 and bear_power_1d_aligned[i] < 0:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_ltf[i]
                    signals[i] = 0.25
                # Short signal: Bear Power > 0 and Bull Power < 0
                elif bear_power_1d_aligned[i] > 0 and bull_power_1d_aligned[i] < 0:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_ltf[i]
                    signals[i] = -0.25
    
    return signals