#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1h volume/price action for timing
# - Signal direction: 4h Donchian(20) breakout (bullish if price > upper band, bearish if price < lower band)
# - Entry filter: 1h volume > 1.3x 20-period average AND price closes in upper/lower 30% of 1h range
# - Regime filter: 1d ADX(14) > 25 to ensure trending market (avoid choppy conditions)
# - Exit: Donchian mean reversion (price crosses middle band) or ATR-based stoploss (2.0x ATR)
# - Position size: 0.20 discrete level to minimize fee churn
# - Session filter: 08-20 UTC to avoid low-volume Asian session
# - Target: 15-25 trades/year (60-100 total over 4 years) - conservative for 1h timeframe
# - Works in bull/bear: Donchian breakouts capture trends, ADX filter avoids false signals in ranging markets

name = "1h_4h_1d_donchian_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian Channel (20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 1h volume filter
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_1h > (1.3 * avg_volume_20)
    
    # Pre-compute 1h price position in range (0=low, 1=high)
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    range_1h = high_1h - low_1h
    # Avoid division by zero
    range_1h = np.where(range_1h == 0, 1e-10, range_1h)
    close_position = (close_1h - low_1h) / range_1h  # 0 at low, 1 at high
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ns], use index.hour if DatetimeIndex, else convert
    if hasattr(prices.index, 'hour'):
        hours = prices.index.hour
    else:
        hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_20[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(close_position[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] < donchian_mid[i] or close_4h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_4h[i] > donchian_mid[i] or close_4h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for entries aligned with 4h Donchian direction
            # Long: price above upper Donchian band with volume and close strength
            if (close_4h[i] > donchian_high[i] and 
                volume_filter[i] and 
                close_position[i] > 0.7 and  # close in upper 30% of 1h range
                adx_aligned[i] > 25):      # trending market
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.20
            # Short: price below lower Donchian band with volume and close weakness
            elif (close_4h[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close_position[i] < 0.3 and  # close in lower 30% of 1h range
                  adx_aligned[i] > 25):      # trending market
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.20
    
    return signals