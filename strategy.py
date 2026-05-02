#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong trends (ADX > 25), we take trend-following entries: long when Bull Power > 0, short when Bear Power < 0
# In ranging markets (ADX < 20), we mean-revert at extremes: long when Bear Power < -std, short when Bull Power > +std
# Volume confirmation (2.0x 20-period average) ensures institutional participation. Designed for 50-150 total trades
# over 4 years (12-37/year) on 6h timeframe. Works in bull markets (trend following) and bear markets
# (mean reversion in ranges) by adapting to ADX regime.

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
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for ADX and EMA13)
    start_idx = max(50, 30)  # 50 bars for ADX, 30 bars for EMA13 stability
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if adx > 25:  # Trending regime - trend following
                # Long: Bull Power > 0 (buying pressure) with volume spike
                if bull_power[i] > 0 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (selling pressure) with volume spike
                elif bear_power[i] < 0 and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif adx < 20:  # Ranging regime - mean reversion
                # Calculate volatility of Bear Power for entry thresholds
                # Use 20-period std of Bear Power
                if i >= 20:
                    bear_power_ma = np.nanmean(bear_power[i-20:i])
                    bear_power_std = np.nanstd(bear_power[i-20:i])
                    bull_power_ma = np.nanmean(bull_power[i-20:i])
                    bull_power_std = np.nanstd(bull_power[i-20:i])
                    
                    # Long: Bear Power < -0.5 * std (oversold) with volume spike
                    if bear_power[i] < (bear_power_ma - 0.5 * bear_power_std) and vol_spike:
                        signals[i] = 0.25
                        position = 1
                    # Short: Bull Power > +0.5 * std (overbought) with volume spike
                    elif bull_power[i] > (bull_power_ma + 0.5 * bull_power_std) and vol_spike:
                        signals[i] = -0.25
                        position = -1
            else:  # Transition regime (20 <= ADX <= 25) - no trade
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions based on regime
            if adx > 25:  # Trending: exit when Bull Power turns negative
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif adx < 20:  # Ranging: exit when Bear Power reverts to mean
                if i >= 20:
                    bear_power_ma = np.nanmean(bear_power[i-20:i])
                    if bear_power[i] >= bear_power_ma:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:  # Transition: exit on any regime change or power reversal
                if adx >= 20 and adx <= 25:  # Still in transition
                    signals[i] = 0.25
                elif adx > 25 and bull_power[i] <= 0:  # Shifted to trending, power failed
                    signals[i] = 0.0
                    position = 0
                elif adx < 20 and bear_power[i] >= np.nanmean(bear_power[max(0,i-20):i]):  # Shifted to ranging, reverted
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions based on regime
            if adx > 25:  # Trending: exit when Bear Power turns positive
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif adx < 20:  # Ranging: exit when Bull Power reverts to mean
                if i >= 20:
                    bull_power_ma = np.nanmean(bull_power[i-20:i])
                    if bull_power[i] <= bull_power_ma:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:  # Transition: exit on any regime change or power reversal
                if adx >= 20 and adx <= 25:  # Still in transition
                    signals[i] = -0.25
                elif adx > 25 and bear_power[i] >= 0:  # Shifted to trending, power failed
                    signals[i] = 0.0
                    position = 0
                elif adx < 20 and bull_power[i] <= np.nanmean(bull_power[max(0,i-20):i]):  # Shifted to ranging, reverted
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals