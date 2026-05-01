#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d ADX > 25 indicates strong trend (trend-following mode), ADX < 20 indicates range (mean-reversion mode)
# In trend regime: enter long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# In range regime: enter long when Bear Power < -0.5*ATR and turning up, short when Bull Power > 0.5*ATR and turning down
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~15-30 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets via regime adaptation

name = "6h_ElderRay_1dADX_Regime_Volume_v1"
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
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h ATR(14) for range regime thresholds
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = 0
    atr_14 = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 13 for EMA13 + 14 for ATR + 14*3 for ADX
    start_idx = max(13, 14, 14*3)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX: ADX > 25 = trend, ADX < 20 = range
        is_trend = adx_aligned[i] > 25
        is_range = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if is_trend:
                # Trend regime: follow Elder Ray momentum
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_range:
                # Range regime: mean reversion at extremes
                if bear_power[i] < -0.5 * atr_14[i] and bear_power[i] > bear_power[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif bull_power[i] > 0.5 * atr_14[i] and bull_power[i] < bull_power[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_trend:
                # Exit trend long when bear power turns positive or momentum fades
                if bear_power[i] > 0 or bull_power[i] < bull_power[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif is_range:
                # Exit range long when bull power approaches zero or turns negative
                if bull_power[i] > -0.2 * atr_14[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Transition regime: exit on any opposite signal
                if bear_power[i] > 0 or bull_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_trend:
                # Exit trend short when bull power turns negative or momentum fades
                if bull_power[i] < 0 or bear_power[i] > bear_power[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif is_range:
                # Exit range short when bear power approaches zero or turns positive
                if bear_power[i] < 0.2 * atr_14[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Transition regime: exit on any opposite signal
                if bull_power[i] < 0 or bear_power[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals