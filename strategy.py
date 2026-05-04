#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure buying/selling pressure
# 1d ADX regime filter: ADX > 25 = trend (trade Elder Ray extremes), ADX < 20 = range (fade Elder Ray extremes)
# Volume confirmation: 6h volume > 1.5x 20 EMA volume ensures institutional participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 80-140 total trades over 4 years = 20-35/year for 6h timeframe
# Elder Ray works in both bull and bear markets by adapting to regime via ADX filter

name = "6h_ElderRay_1dADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
    minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Shift ADX by 1 to use only prior completed 1d bar
    adx_shifted = np.roll(adx, 1)
    adx_shifted[0] = np.nan
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_shifted)
    
    # 6h EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray components
    bull_power = high - ema13   # Buying pressure
    bear_power = low - ema13    # Selling pressure (negative values)
    
    # 6h Volume confirmation: 20 EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry logic
            if adx_1d_aligned[i] > 25:  # Trending regime - trade with Elder Ray extremes
                # Long: strong buying pressure + volume confirmation
                if bull_power[i] > 0 and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: strong selling pressure + volume confirmation
                elif bear_power[i] < 0 and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = -0.25
                    position = -1
            elif adx_1d_aligned[i] < 20:  # Range regime - fade Elder Ray extremes
                # Long: selling exhaustion (bear power weakening) + volume confirmation
                if bear_power[i] < 0 and bear_power[i] > bear_power[i-1] and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: buying exhaustion (bull power weakening) + volume confirmation
                elif bull_power[i] > 0 and bull_power[i] < bull_power[i-1] and volume[i] > (1.5 * vol_ema_20[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Elder Ray weakening OR regime change to ranging
            if (bull_power[i] < 0) or (adx_1d_aligned[i] < 20 and bear_power[i] > bear_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray weakening OR regime change to ranging
            if (bear_power[i] > 0) or (adx_1d_aligned[i] < 20 and bull_power[i] < bull_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals