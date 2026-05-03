#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation.
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish imbalance) in non-trending market (1d ADX < 25) with volume > 1.5x 20-period MA.
# Short when Bear Power > 0 AND Bull Power < 0 (bearish imbalance) in non-trending market (1d ADX < 25) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining sufficient exposure.
# 1d ADX < 25 identifies ranging/low-volatility environments where Elder Ray works best for mean reversion.
# Volume confirmation ensures moves have participation, reducing false signals in chop.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "6h_ElderRay_1dADX25_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:  # ADX needs at least 14 periods
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Wilder's smoothing: result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Index
    bull_power = high - ema_13  # High - EMA
    bear_power = ema_13 - low   # EMA - Low
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Regime: non-trending market (ADX < 25) where Elder Ray works best for mean reversion
        is_ranging = adx_val < 25
        
        # Elder Ray conditions
        is_bullish_imbalance = bp > 0 and br < 0  # Bull Power positive AND Bear Power negative
        is_bearish_imbalance = br > 0 and bp < 0  # Bear Power positive AND Bull Power negative
        
        # Entry logic
        if position == 0:
            if is_ranging and is_bullish_imbalance and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_ranging and is_bearish_imbalance and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: imbalance reverses OR ADX trends up (trending market)
            if not is_bullish_imbalance or adx_val >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: imbalance reverses OR ADX trends up (trending market)
            if not is_bearish_imbalance or adx_val >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals