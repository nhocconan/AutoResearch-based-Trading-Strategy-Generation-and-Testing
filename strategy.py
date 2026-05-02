#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d ADX Regime Filter with Volume Confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# In strong trends (ADX > 25), trade with the trend: long when Bull Power > 0, short when Bear Power > 0
# In ranging markets (ADX < 20), mean revert at extremes: long when Bear Power < 0 and price near low, short when Bull Power < 0 and price near high
# Volume confirmation filters weak breakouts. Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull markets by trend-following strength, in bear markets by trend-following weakness and mean reversion in ranges.

name = "6h_ElderRay_Power_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 30)  # EMA13 + ADX warmup
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if adx_val > 25:  # Trending regime
                # Long: Bull Power > 0 with volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 with volume spike
                elif bear_power[i] > 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (ADX < 25)
                # Long: Bear Power < 0 (bullish bias) and price near low
                if bear_power[i] < 0 and close[i] <= low[i] * 1.005:  # within 0.5% of low
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power < 0 (bearish bias) and price near high
                elif bull_power[i] < 0 and close[i] >= high[i] * 0.995:  # within 0.5% of high
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if adx_val > 25:  # Trending: exit when bull power fades
                if bull_power[i] <= 0:
                    exit_long = True
            else:  # Ranging: exit when price moves to middle or bear power turns positive
                if close[i] >= (high[i] + low[i]) / 2 or bear_power[i] > 0:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if adx_val > 25:  # Trending: exit when bear power fades
                if bear_power[i] <= 0:
                    exit_short = True
            else:  # Ranging: exit when price moves to middle or bull power turns positive
                if close[i] <= (high[i] + low[i]) / 2 or bull_power[i] > 0:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals