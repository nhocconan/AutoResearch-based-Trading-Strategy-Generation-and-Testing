#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(10) breakout with 1w ADX filter and volume confirmation
# Uses shorter Donchian period for more frequent signals while maintaining quality
# ADX > 25 ensures we only trade in trending markets, reducing whipsaws
# Volume > 20-period average confirms breakout strength
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets
# Shorter Donchian period increases trade frequency but ADX filter maintains quality

name = "1d_donchian10_1w_adx_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on 1w data
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth(val, period):
        return np.convolve(val, np.ones(period)/period, mode='same')
    
    atr = np.zeros_like(tr)
    atr[14:] = np.convolve(tr[14:], np.ones(14)/14, mode='valid')[:len(atr)-14]
    atr = np.concatenate([np.full(14, np.nan), atr[14:]])
    
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    dm_plus_smooth[14:] = np.convolve(dm_plus[14:], np.ones(14)/14, mode='valid')[:len(dm_plus)-14]
    dm_minus_smooth[14:] = np.convolve(dm_minus[14:], np.ones(14)/14, mode='valid')[:len(dm_minus)-14]
    dm_plus_smooth = np.concatenate([np.full(14, np.nan), dm_plus_smooth[14:]])
    dm_minus_smooth = np.concatenate([np.full(14, np.nan), dm_minus_smooth[14:]])
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[27:] = np.convolve(dx[27:], np.ones(14)/14, mode='valid')[:len(dx)-27]
    adx = np.concatenate([np.full(27, np.nan), adx[27:]])
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian channels (10-period for more signals)
    high_max = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or ADX < 20 (trend weakening)
            elif close[i] < low_min[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or ADX < 20 (trend weakening)
            elif close[i] > high_max[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and strong trend (ADX > 25)
            if vol_filter[i] and adx_aligned[i] > 25:
                # Long when price breaks above Donchian high
                if close[i] > high_max[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price breaks below Donchian low
                elif close[i] < low_min[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals