#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend filter (price vs Kumo cloud) and TK cross timing.
- Entry: Long when price > Senkou Span A/B (cloud top) AND TK cross bullish AND volume > 1.5 * 6h volume MA(20);
         Short when price < Senkou Span A/B (cloud bottom) AND TK cross bearish AND volume > 1.5 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via cloud (signal=0 when price re-enters cloud).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Ichimoku cloud provides dynamic support/resistance; TK cross gives momentum timing.
- Works in bull markets via trend continuation and bear markets via cloud-filtered mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # TK Cross (Tenkan-sen/Kijun-sen cross)
    # Bullish TK: Tenkan-sen crosses above Kijun-sen
    # Bearish TK: Tenkan-sen crosses below Kijun-sen
    tk_bullish = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_bearish = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    
    # Get 6h data for volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    # Ichimoku components need additional_delay_bars=1 for 1d alignment (completed 1d candle)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top, additional_delay_bars=1)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom, additional_delay_bars=1)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish.astype(float), additional_delay_bars=1)
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish.astype(float), additional_delay_bars=1)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 52  # Ichimoku needs 52 periods for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price re-enters the cloud (mean reversion signal)
        if position == 1:
            if curr_close <= cloud_top_aligned[i] and curr_close >= cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close <= cloud_top_aligned[i] and curr_close >= cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation
        price_above_cloud = curr_close > cloud_top_aligned[i]
        price_below_cloud = curr_close < cloud_bottom_aligned[i]
        
        tk_bull = tk_bullish_aligned[i] > 0.5
        tk_bear = tk_bearish_aligned[i] > 0.5
        
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price above cloud AND bullish TK cross
                if price_above_cloud and tk_bull:
                    signals[i] = 0.25
                    position = 1
                # Short: price below cloud AND bearish TK cross
                elif price_below_cloud and tk_bear:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0