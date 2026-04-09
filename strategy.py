#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1-day timeframe for support/resistance,
# volume confirmation for conviction, and choppiness index regime filter to avoid ranging markets.
# Long when price touches or breaks above Camarilla H3 level with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Short when price touches or breaks below Camarilla L3 level with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Exit when price returns to Camarilla H4/L4 levels (mean reversion) or chop > 61.8 (ranging).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: Camarilla levels adapt to volatility, volume confirms institutional interest,
# chop filter avoids whipsaws in ranging markets during bear regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Camarilla: based on previous day's high, low, close
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), H2 = close + 0.7*(high-low), H1 = close + 0.5*(high-low)
    # L1 = close - 0.5*(high-low), L2 = close - 0.7*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    df_1d = df_1d.copy()
    df_1d['cam_h4'] = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    df_1d['cam_h3'] = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    df_1d['cam_l3'] = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    df_1d['cam_l4'] = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    cam_h4_1d = df_1d['cam_h4'].values
    cam_h3_1d = df_1d['cam_h3'].values
    cam_l3_1d = df_1d['cam_l3'].values
    cam_l4_1d = df_1d['cam_l4'].values
    
    cam_h4 = align_htf_to_ltf(prices, df_1d, cam_h4_1d)
    cam_h3 = align_htf_to_ltf(prices, df_1d, cam_h3_1d)
    cam_l3 = align_htf_to_ltf(prices, df_1d, cam_l3_1d)
    cam_l4 = align_htf_to_ltf(prices, df_1d, cam_l4_1d)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(cam_h3[i]) or np.isnan(cam_l3[i]) or np.isnan(cam_h4[i]) or np.isnan(cam_l4[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price returns to H4 level (mean reversion) or chop > 61.8 (ranging market)
            if close[i] <= cam_h4[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to L4 level (mean reversion) or chop > 61.8 (ranging market)
            if close[i] >= cam_l4[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Camarilla level touch/break with volume and regime confirmation
            bullish_break = (high[i] >= cam_h3[i]) and volume_confirmed and trending_market
            bearish_break = (low[i] <= cam_l3[i]) and volume_confirmed and trending_market
            
            if bullish_break:
                position = 1
                signals[i] = 0.25
            elif bearish_break:
                position = -1
                signals[i] = -0.25
    
    return signals