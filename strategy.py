#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + 1d volume spike + 1w ADX trend filter
# - Primary signal: Price touches Camarilla H3/L3 levels (mean reversion in range)
# - Volume confirmation: 1d volume > 1.8x 20-period average (avoid fakeouts)
# - Regime filter: 1w ADX > 25 (trending market) enables breakout continuation
# - Works in bull/bear: In ranges (ADX < 25), fade H3/L3 touches; in trends (ADX > 25), break Donchian(20)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "4h_1d_1w_camarilla_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / np.where(tr_14 == 0, 1, tr_14)
    di_minus = 100 * dm_minus_14 / np.where(tr_14 == 0, 1, tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # Pre-compute 4h Camarilla levels (based on previous day)
    # Camarilla: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    # We need daily OHLC from 1d data, aligned to 4h
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    camarilla_h3 = align_htf_to_ltf(prices, df_1d, daily_close + (daily_high - daily_low) * 1.1 / 4)
    camarilla_l3 = align_htf_to_ltf(prices, df_1d, daily_close - (daily_high - daily_low) * 1.1 / 4)
    camarilla_h4 = align_htf_to_ltf(prices, df_1d, daily_close + (daily_high - daily_low) * 1.1 / 2)
    camarilla_l4 = align_htf_to_ltf(prices, df_1d, daily_close - (daily_high - daily_low) * 1.1 / 2)
    
    # Pre-compute 4h Donchian(20) for breakout signals
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(20) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_20 = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            # Mean reversion: price reaches camarilla h3/h4 in ranging market
            if adx_aligned[i] < 25 and close_4h[i] >= camarilla_h3[i]:
                exit_signal = True
            # Breakout continuation: price breaks donchian high in trending market
            elif adx_aligned[i] >= 25 and close_4h[i] > donchian_high[i]:
                exit_signal = True  # Let winner run, but we'll exit on mean reversion of camarilla
            # Stoploss
            elif close_4h[i] < entry_price - 2.0 * atr_20[i]:
                exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            # Mean reversion: price reaches camarilla l3/l4 in ranging market
            if adx_aligned[i] < 25 and close_4h[i] <= camarilla_l3[i]:
                exit_signal = True
            # Breakout continuation: price breaks donchian low in trending market
            elif adx_aligned[i] >= 25 and close_4h[i] < donchian_low[i]:
                exit_signal = True
            # Stoploss
            elif close_4h[i] > entry_price + 2.0 * atr_20[i]:
                exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries based on regime
            # Ranging market (ADX < 25): mean reversion at camarilla h3/l3
            if adx_aligned[i] < 25:
                if volume_spike_aligned[i]:
                    # Long: price touches camarilla l3
                    if close_4h[i] <= camarilla_l3[i] * 1.001:  # small buffer
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price touches camarilla h3
                    elif close_4h[i] >= camarilla_h3[i] * 0.999:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
            # Trending market (ADX >= 25): breakout of donchian(20)
            else:
                if volume_spike_aligned[i]:
                    # Long: price breaks above donchian high
                    if close_4h[i] > donchian_high[i]:
                        position = 1
                        entry_price = close_4h[i]
                        signals[i] = 0.25
                    # Short: price breaks below donchian low
                    elif close_4h[i] < donchian_low[i]:
                        position = -1
                        entry_price = close_4h[i]
                        signals[i] = -0.25
    
    return signals