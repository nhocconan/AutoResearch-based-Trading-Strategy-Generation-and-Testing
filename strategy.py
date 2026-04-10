#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels for mean reversion in ranging markets
# - Uses 1w Camarilla levels (H3/L3, H4/L4) as key support/resistance from weekly structure
# - Enters long near weekly L3 (75% retracement of weekly range) with bullish 6h candle close
# - Enters short near weekly H3 with bearish 6h candle close
# - Volume confirmation: current 6h volume > 1.5x 20-period average to avoid false breakouts
# - Regime filter: 1d ADX < 25 to ensure ranging market (avoid strong trends where mean reversion fails)
# - Designed for 6h timeframe: targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Works in ranging markets which frequently occur in BTC/ETH during consolidation periods
# - Uses discrete position sizing (0.25) to minimize fee churn from frequent signal changes

name = "6h_1w_camarilla_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w Camarilla pivot levels (based on prior week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate prior week's Camarilla levels (shifted by 1 to avoid look-ahead)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # H2 = close + 0.55*(high-low), L2 = close - 0.55*(high-low)
    # H1 = close + 0.275*(high-low), L1 = close - 0.275*(high-low)
    # Pivot = (high + low + close)/3
    
    weekly_range = high_1w - low_1w
    camarilla_h4 = close_1w + 1.5 * weekly_range
    camarilla_l4 = close_1w - 1.5 * weekly_range
    camarilla_h3 = close_1w + 1.1 * weekly_range
    camarilla_l3 = close_1w - 1.1 * weekly_range
    camarilla_h2 = close_1w + 0.55 * weekly_range
    camarilla_l2 = close_1w - 0.55 * weekly_range
    camarilla_h1 = close_1w + 0.275 * weekly_range
    camarilla_l1 = close_1w - 0.275 * weekly_range
    camarilla_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Pre-compute 1d ADX(14) for regime filter (ranging market: ADX < 25)
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
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for mean reversion entries
            # Only trade in ranging markets (ADX < 25)
            if adx_aligned[i] < 25 and vol_spike[i]:
                close_price = prices['close'].iloc[i]
                
                # Long entry: price near weekly L3 (75% retracement) with bullish close
                if close_price <= l3_aligned[i] * 1.005 and close_price > prices['open'].iloc[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price near weekly H3 (75% retracement) with bearish close
                elif close_price >= h3_aligned[i] * 0.995 and close_price < prices['open'].iloc[i]:
                    position = -1
                    signals[i] = -0.25
        
        elif position == 1:  # Long position - exit at mean reversion targets
            close_price = prices['close'].iloc[i]
            # Exit: price reaches weekly H3 (mean reversion target) or weekly L4 (stop)
            if close_price >= h3_aligned[i] * 0.995 or close_price <= l4_aligned[i] * 1.005:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position - exit at mean reversion targets
            close_price = prices['close'].iloc[i]
            # Exit: price reaches weekly L3 (mean reversion target) or weekly H4 (stop)
            if close_price <= l3_aligned[i] * 1.005 or close_price >= h4_aligned[i] * 0.995:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals