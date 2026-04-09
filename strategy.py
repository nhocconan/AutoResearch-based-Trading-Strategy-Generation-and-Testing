#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d regime filter and volume confirmation
# - Uses 4h EMA(50) for trend direction and 1d ADX(14) for trend strength
# - Long when: price > 4h EMA50, 1d ADX > 25, and price breaks above H3 Camarilla pivot level with volume > 1.5x 20-period average
# - Short when: price < 4h EMA50, 1d ADX > 25, and price breaks below L3 Camarilla pivot level with volume > 1.5x 20-period average
# - Fixed position size 0.20 to control drawdown
# - Session filter: only trade 08-20 UTC to avoid low-volume Asian session
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years)
# - Camarilla pivots provide precise support/resistance levels that work in ranging markets
# - ADX filter ensures we only trade in trending conditions, reducing false breakouts
# - Volume confirmation adds validity to breakouts
# - Works in both bull and bear markets by capturing breakouts in trending phases

name = "1h_4h_1d_camarilla_pivot_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Camarilla pivot levels (based on previous day's range)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First bar uses current close as previous
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    h3 = prev_close_1d + 1.25 * camarilla_range
    l3 = prev_close_1d - 1.25 * camarilla_range
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0 or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit if price falls below EMA50 or ADX weakens
            if close[i] < ema_50_4h_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if price rises above EMA50 or ADX weakens
            if close[i] > ema_50_4h_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic: Camarilla breakout + trend filter + volume confirmation
            if volume_confirmed and adx_aligned[i] > 25:
                # Long entry: price breaks above H3 with uptrend
                if close[i] > h3_aligned[i] and close[i] > ema_50_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price breaks below L3 with downtrend
                elif close[i] < l3_aligned[i] and close[i] < ema_50_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals