#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span) on 6h
# - Long when price > cloud AND Tenkan > Kijun (bullish TK cross) AND 1d ADX > 25
# - Short when price < cloud AND Tenkan < Kijun (bearish TK cross) AND 1d ADX > 25
# - Volume confirmation: current volume > 1.5x 20-period 6h volume average
# - Exit when TK cross reverses or price enters cloud
# - Ichimoku works in all markets: cloud acts as dynamic S/R, TK cross captures momentum
# - 1d ADX filter ensures we only trade when higher timeframe is trending
# - Volume confirmation prevents false breakouts in low participation
# - Target: 12-25 trades/year on 6h (50-100 total over 4 years)

name = "6h_1d_ichimoku_tk_adx_volume_v1"
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
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_9 = np.full_like(high_6h, np.nan, dtype=float)
    lowest_low_9 = np.full_like(low_6h, np.nan, dtype=float)
    for i in range(period_tenkan - 1, len(high_6h)):
        highest_high_9[i] = np.max(high_6h[i - period_tenkan + 1:i + 1])
        lowest_low_9[i] = np.min(low_6h[i - period_tenkan + 1:i + 1])
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_26 = np.full_like(high_6h, np.nan, dtype=float)
    lowest_low_26 = np.full_like(low_6h, np.nan, dtype=float)
    for i in range(period_kijun - 1, len(high_6h)):
        highest_high_26[i] = np.max(high_6h[i - period_kijun + 1:i + 1])
        lowest_low_26[i] = np.min(low_6h[i - period_kijun + 1:i + 1])
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    highest_high_52 = np.full_like(high_6h, np.nan, dtype=float)
    lowest_low_52 = np.full_like(low_6h, np.nan, dtype=float)
    for i in range(period_senkou_b - 1, len(high_6h)):
        highest_high_52[i] = np.max(high_6h[i - period_senkou_b + 1:i + 1])
        lowest_low_52[i] = np.min(low_6h[i - period_senkou_b + 1:i + 1])
    senkou_b = (highest_high_52 + lowest_low_52) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou = np.full_like(close_6h, np.nan, dtype=float)
    for i in range(len(close_6h) - 26):
        chikou[i + 26] = close_6h[i]
    
    # Pre-compute 1d ADX(14)
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
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder smoothing)
    tr_14 = np.full_like(tr, np.nan, dtype=float)
    dm_plus_14 = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_14 = np.full_like(dm_minus, np.nan, dtype=float)
    
    if len(tr) >= 14:
        # Initial values (simple average)
        tr_14[13] = np.nanmean(tr[1:14])
        dm_plus_14[13] = np.nanmean(dm_plus[1:14])
        dm_minus_14[13] = np.nanmean(dm_minus[1:14])
        
        # Wilder smoothing
        for i in range(14, len(tr)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full_like(tr_14, np.nan, dtype=float)
    di_minus = np.full_like(tr_14, np.nan, dtype=float)
    mask = ~np.isnan(tr_14) & (tr_14 != 0)
    di_plus[mask] = (dm_plus_14[mask] / tr_14[mask]) * 100
    di_minus[mask] = (dm_minus_14[mask] / tr_14[mask]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    mask_dx = (~np.isnan(di_plus) & ~np.isnan(di_minus) & 
               ((di_plus + di_minus) != 0))
    dx[mask_dx] = (np.abs(di_plus[mask_dx] - di_minus[mask_dx]) / 
                   (di_plus[mask_dx] + di_minus[mask_dx])) * 100
    
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        # Initial ADX (simple average of first 14 DX)
        valid_dx = dx[14:28]  # indices 14 to 27
        if not np.all(np.isnan(valid_dx)):
            adx[27] = np.nanmean(valid_dx)
            # Wilder smoothing for ADX
            for i in range(28, len(dx)):
                if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_tenkan = np.full(n, np.nan)  # for TK cross detection
    prev_kijun = np.full(n, np.nan)
    
    for i in range(100, n):  # Start after warmup
        # Store previous Tenkan and Kijun for crossover detection
        if i > 0:
            prev_tenkan[i] = tenkan[i-1]
            prev_kijun[i] = kijun[i-1]
        else:
            prev_tenkan[i] = np.nan
            prev_kijun[i] = np.nan
        
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(chikou[i]) or np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_ma_6h = np.full_like(vol_series, np.nan, dtype=float)
        for j in range(19, i+1):
            vol_ma_6h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_6h[i]) and vol_series[i] > 1.5 * vol_ma_6h[i]
        
        close_price = close_6h[i]
        tenkan_now = tenkan[i]
        kijun_now = kijun[i]
        tenkan_prev = prev_tenkan[i]
        kijun_prev = prev_kijun[i]
        
        # Determine cloud boundaries (Senkou Span A/B shifted forward)
        # For current price, we need Senkou values from 26 periods ago
        senkou_a_now = senkou_a[i-26] if i >= 26 else np.nan
        senkou_b_now = senkou_b[i-26] if i >= 26 else np.nan
        
        if i >= 26 and not np.isnan(senkou_a_now) and not np.isnan(senkou_b_now):
            cloud_top = max(senkou_a_now, senkou_b_now)
            cloud_bottom = min(senkou_a_now, senkou_b_now)
            price_above_cloud = close_price > cloud_top
            price_below_cloud = close_price < cloud_bottom
            price_in_cloud = (close_price >= cloud_bottom) and (close_price <= cloud_top)
        else:
            price_above_cloud = False
            price_below_cloud = False
            price_in_cloud = True  # conservative: treat as in cloud if not enough data
        
        # TK cross signals
        tk_bullish_cross = (tenkan_prev <= kijun_prev) and (tenkan_now > kijun_now)
        tk_bearish_cross = (tenkan_prev >= kijun_prev) and (tenkan_now < kijun_now)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > cloud AND bullish TK cross AND 1d trending (ADX > 25) AND volume spike
            if (price_above_cloud and tk_bullish_cross and 
                adx_1d_aligned[i] > 25 and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < cloud AND bearish TK cross AND 1d trending (ADX > 25) AND volume spike
            elif (price_below_cloud and tk_bearish_cross and 
                  adx_1d_aligned[i] > 25 and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: TK cross reverses OR price enters cloud
            exit_long = (position == 1 and (tk_bearish_cross or price_in_cloud))
            exit_short = (position == -1 and (tk_bullish_cross or price_in_cloud))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals