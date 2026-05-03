#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime + volume confirmation
# Bull power = high - EMA(13), Bear power = EMA(13) - low
# Regime: ADX(14) > 25 = trending, ADX < 20 = ranging (hysteresis)
# In trending regime: follow Elder Ray (long if bull power > 0, short if bear power > 0)
# In ranging regime: mean revert at extremes (long if bull power < -threshold, short if bear power < -threshold)
# Volume confirmation requires 1.5x 20-period average
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag
# Works in both bull and bear markets by adapting to regime

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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMA(13) on 6h for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation (1.5x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    regime_hysteresis = 0  # 0: undefined, 1: trending, -1: ranging
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(13 for EMA, 20 for volume MA +1 for shift, 30 for ADX)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime with hysteresis
        if regime_hysteresis == 0:  # Initial determination
            if adx_aligned[i] > 25:
                regime_hysteresis = 1  # Trending
            elif adx_aligned[i] < 20:
                regime_hysteresis = -1  # Ranging
        elif regime_hysteresis == 1:  # Currently trending
            if adx_aligned[i] < 20:
                regime_hysteresis = -1  # Switch to ranging
        elif regime_hysteresis == -1:  # Currently ranging
            if adx_aligned[i] > 25:
                regime_hysteresis = 1  # Switch to trending
        
        if position == 0:  # Flat - look for new entries
            if regime_hysteresis == 1:  # Trending regime
                # Long: bull power positive + volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bear power positive + volume spike
                elif bear_power[i] > 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif regime_hysteresis == -1:  # Ranging regime
                # Long: bull power significantly negative (oversold) + volume spike
                if bull_power[i] < -0.5 * np.std(bull_power[max(0, i-50):i+1]) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: bear power significantly negative (overbought) + volume spike
                elif bear_power[i] < -0.5 * np.std(bear_power[max(0, i-50):i+1]) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if regime_hysteresis == 1:  # Trending regime
                # Exit if bull power turns negative (trend weakness)
                if bull_power[i] <= 0:
                    exit_signal = True
            elif regime_hysteresis == -1:  # Ranging regime
                # Exit if bull power returns to zero (mean reversion complete)
                if bull_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if regime_hysteresis == 1:  # Trending regime
                # Exit if bear power turns negative (trend weakness)
                if bear_power[i] <= 0:
                    exit_signal = True
            elif regime_hysteresis == -1:  # Ranging regime
                # Exit if bear power returns to zero (mean reversion complete)
                if bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals