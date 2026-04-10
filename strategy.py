#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Regime filter: 1d ADX > 25 for trending, < 20 for ranging (hysteresis)
# - In trending regime (ADX > 25): trend follow - long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging regime (ADX < 20): mean revert - long when Bull Power < 0 and price < Donchian(20) low, short when Bear Power < 0 and price > Donchian(20) high
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)
# - Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets

name = "6h_1d_elder_ray_regime_volume_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for regime filter
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ADX(14) for regime detection with hysteresis
    # Calculate +DM, -DM, TR
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14_1d = WilderSmoothing(tr, 14)
    plus_di_14_1d = 100 * WilderSmoothing(plus_dm, 14) / atr_14_1d
    minus_di_14_1d = 100 * WilderSmoothing(minus_dm, 14) / atr_14_1d
    dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d + 1e-10)
    adx_14_1d = WilderSmoothing(dx_14_1d, 14)
    
    # Regime states with hysteresis: 0=ranging (ADX<20), 1=trending (ADX>25), -1=hold previous
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Pre-compute 6h indicators
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_6h = high_6h - ema_13_6h
    bear_power_6h = ema_13_6h - low_6h
    
    # 6h volume confirmation: > 1.5x 20-period average
    avg_volume_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume_6h > (1.5 * avg_volume_20_6h)
    
    # 6h ATR(14) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14_6h = WilderSmoothing(tr_6h, 14)
    
    # 6h Donchian(20) for mean reversion signals
    donchian_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    regime = 0  # 0=ranging, 1=trending, -1=hold
    
    for i in range(50, n):
        # Update regime with hysteresis
        if not np.isnan(adx_14_1d_aligned[i]):
            if adx_14_1d_aligned[i] > 25:
                regime = 1  # trending
            elif adx_14_1d_aligned[i] < 20:
                regime = 0  # ranging
            # else hold previous regime
        
        # Skip if any required data is invalid
        if (np.isnan(ema_13_6h[i]) or np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(vol_spike_6h[i]) or np.isnan(atr_14_6h[i]) or np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i])):
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
            # Look for entry signals based on regime
            if vol_spike_6h[i]:
                if regime == 1:  # Trending regime - trend follow
                    # Long: Bull Power positive and rising (more bullish)
                    if (bull_power_6h[i] > 0 and i > 50 and bull_power_6h[i] > bull_power_6h[i-1]):
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_6h[i]
                        signals[i] = 0.25
                    # Short: Bear Power positive and rising (more bearish)
                    elif (bear_power_6h[i] > 0 and i > 50 and bear_power_6h[i] > bear_power_6h[i-1]):
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_6h[i]
                        signals[i] = -0.25
                elif regime == 0:  # Ranging regime - mean revert
                    # Long: Bull Power negative (weak bulls) and price at support
                    if (bull_power_6h[i] < 0 and prices['close'].iloc[i] <= donchian_low_6h[i]):
                        position = 1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_6h[i]
                        signals[i] = 0.25
                    # Short: Bear Power negative (weak bears) and price at resistance
                    elif (bear_power_6h[i] < 0 and prices['close'].iloc[i] >= donchian_high_6h[i]):
                        position = -1
                        entry_price = prices['close'].iloc[i]
                        entry_atr = atr_14_6h[i]
                        signals[i] = -0.25
    
    return signals