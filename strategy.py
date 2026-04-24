#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike filter and 1d ADX trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume spike and ADX trend filter.
- Entry: Long when price breaks above Camarilla H3 AND volume > 1.5x 20-period average volume AND ADX > 25.
         Short when price breaks below Camarilla L3 AND volume > 1.5x 20-period average volume AND ADX > 25.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Volume confirmation avoids low-volume false breakouts.
- ADX > 25 ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on strict confluence of breakout, volume, and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / (tr_period + 1e-10)
    di_minus = 100 * dm_minus_period / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX for trend filter
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    
    # 1d volume average for volume spike filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 12h primary timeframe
    # Camarilla levels based on previous day's OHLC
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    # We need to calculate these using daily OHLC but apply to 12h timeframe
    
    # For 12h timeframe, we'll use rolling window of 2 bars (since 2*12h = 24h ~ 1 day)
    # But better approach: calculate daily OHLC from 1d data and align to 12h
    # However, since we're on 12h timeframe, we can approximate using 2-period window
    
    # Calculate typical price for Camarilla calculation
    typical_price = (high + low + close) / 3
    
    # Calculate Camarilla H3 and L3 using 2-period lookback (approx 1 day for 12h TF)
    # H3 = typical_price + 1.1 * (high - low) / 2
    # L3 = typical_price - 1.1 * (high - low) / 2
    high_low_diff = high - low
    camarilla_h3 = typical_price + 1.1 * high_low_diff / 2
    camarilla_l3 = typical_price - 1.1 * high_low_diff / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma if vol_ma > 0 else False
        
        # ADX trend condition: ADX > 25 indicates trending market
        strong_trend = adx_1d_aligned[i] > 25
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below Camarilla L3
            if position == 1:
                if curr_close < camarilla_l3[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3
            elif position == -1:
                if curr_close > camarilla_h3[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume and trend confirmation
        if position == 0:
            # Long: price breaks above Camarilla H3 AND volume spike AND strong trend
            if curr_close > camarilla_h3[i] and volume_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND volume spike AND strong trend
            elif curr_close < camarilla_l3[i] and volume_spike and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1dADX_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0