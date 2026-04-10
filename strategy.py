#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross + 1d ADX Trend Filter
# - Uses Ichimoku (Tenkan/Kijun/Senkou Span A/B) on 6h for trend and momentum
# - Long when: TK Cross bullish AND price > Cloud AND 1d ADX > 25 (strong trend)
# - Short when: TK Cross bearish AND price < Cloud AND 1d ADX > 25 (strong trend)
# - Exit when TK Cross reverses OR price re-enters Cloud
# - ADX filter ensures we only trade in strong trending markets (works in bull/bear)
# - Ichimoku provides dynamic support/resistance via Cloud
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_ichimoku_tk_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used in signals as it requires future data
    
    # Ichimoku Cloud: between Senkou Span A and B
    # Cloud top = max(Senkou A, Senkou B)
    # Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross: Tenkan crossing above/below Kijun
    tk_cross = tenkan - kijun  # >0 = bullish cross, <0 = bearish cross
    
    # Price relative to Cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Pre-compute 1d ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    # Initialize arrays
    tr_smoothed = np.full_like(tr, np.nan)
    plus_dm_smoothed = np.full_like(plus_dm, np.nan)
    minus_dm_smoothed = np.full_like(minus_dm, np.nan)
    
    # First value is simple average
    if len(tr) >= atr_period:
        tr_smoothed[atr_period-1] = np.nansum(tr[:atr_period])
        plus_dm_smoothed[atr_period-1] = np.nansum(plus_dm[:atr_period])
        minus_dm_smoothed[atr_period-1] = np.nansum(minus_dm[:atr_period])
        
        # Subsequent values: Wilder's smoothing
        for i in range(atr_period, len(tr)):
            tr_smoothed[i] = tr_smoothed[i-1] - (tr_smoothed[i-1] / atr_period) + tr[i]
            plus_dm_smoothed[i] = plus_dm_smoothed[i-1] - (plus_dm_smoothed[i-1] / atr_period) + plus_dm[i]
            minus_dm_smoothed[i] = minus_dm_smoothed[i-1] - (minus_dm_smoothed[i-1] / atr_period) + minus_dm[i]
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = np.zeros_like(tr)
    mask = (plus_di + minus_di) != 0
    dx[mask] = np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask]) * 100
    
    # ADX = smoothed DX
    adx = np.full_like(tr, np.nan)
    if len(dx) >= atr_period:
        # First ADX is average of first 'atr_period' DX values
        adx[2*atr_period-2] = np.nanmean(dx[atr_period-1:2*atr_period-1])
        # Subsequent ADX values: Wilder's smoothing of DX
        for i in range(2*atr_period-1, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1] / atr_period) + dx[i]
    
    # ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    # Align HTF indicators to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup (max Ichimoku period)
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(strong_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: TK Cross bullish AND price > Cloud AND strong 1d trend
            if (tk_cross[i] > 0 and 
                price_above_cloud[i] and 
                strong_trend_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: TK Cross bearish AND price < Cloud AND strong 1d trend
            elif (tk_cross[i] < 0 and 
                  price_below_cloud[i] and 
                  strong_trend_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when TK Cross reverses OR price re-enters Cloud
            exit_long = (position == 1 and 
                        (tk_cross[i] <= 0 or not price_above_cloud[i]))
            exit_short = (position == -1 and 
                         (tk_cross[i] >= 0 or not price_below_cloud[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals