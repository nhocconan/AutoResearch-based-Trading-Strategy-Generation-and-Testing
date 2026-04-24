#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + TK Cross + 1d ADX Regime Filter + Volume Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX for regime filter (ADX > 25 = trending, ADX < 20 = ranging).
- Ichimoku components calculated on 6h: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement).
- Entry: Long when TK cross (Tenkan > Kijun) AND price > Kumo (cloud) AND 1d ADX > 25 AND volume > 1.5 * 6h volume MA(20);
         Short when TK cross (Tenkan < Kijun) AND price < Kumo AND 1d ADX > 25 AND volume > 1.5 * 6h volume MA(20).
- Exit: Long exits when TK cross reverses (Tenkan < Kijun) OR price < Kumo; Short exits when TK cross reverses OR price > Kumo.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying TK cross above cloud in uptrend regime) and bear (selling TK cross below cloud in downtrend regime).
- Volume confirmation adds conviction; ADX regime filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value: simple average
            result[period-1] = np.nanmean(data[:period])
            # Rest: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Ichimoku on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used in signals as it requires future data
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For signaling at time t, we use Senkou A/B from t-26 (already published)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))  # bullish cross
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))  # bearish cross
    
    # Price relative to Kumo
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Volume confirmation: 1.5x threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    # Regime filters from 1d ADX
    adx_trending = adx_aligned > 25
    adx_ranging = adx_aligned < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 20)  # Senkou B needs 52, Kijun needs 26, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals
            # Long: TK cross up AND price above Kumo AND trending regime AND volume confirmation
            if tk_cross_up[i] and price_above_kumo[i] and adx_trending[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND price below Kumo AND trending regime AND volume confirmation
            elif tk_cross_down[i] and price_below_kumo[i] and adx_trending[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when TK cross reverses down OR price falls below Kumo
            if tk_cross_down[i] or price_below_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross reverses up OR price rises above Kumo
            if tk_cross_up[i] or price_above_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0