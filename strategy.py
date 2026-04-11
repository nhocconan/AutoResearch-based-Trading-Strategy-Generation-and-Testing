#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly OHLC for ATR calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR for volatility filter
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly volatility percentile (14-period ATR % rank)
    atr_percentile = pd.Series(atr_w).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Shift to avoid look-ahead
    atr_percentile = np.roll(atr_percentile, 1)
    atr_percentile[0] = np.nan
    
    # Align weekly volatility percentile to 12h timeframe
    vol_percentile_12h = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Weekly OHLC for Camarilla pivot calculation
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
    camarilla_h3 = close_1w + range_1w * 1.1 / 4
    camarilla_l3 = close_1w - range_1w * 1.1 / 4
    
    # Shift by 1 to use only completed weekly bars (previous week's levels)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Align weekly Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX for trend strength (14 period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = tr_w[1:]  # Use weekly TR for consistency
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i]) or np.isnan(vol_percentile_12h[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Volatility filter: trade only when weekly ATR percentile > 0.3 (avoid low volatility)
        vol_filter = vol_percentile_12h[i] > 0.3
        
        # Trend filter: ADX > 25 (strong trend)
        trend_filter = adx[i] > 25
        
        # Long conditions: price breaks above H3 level with volume, volatility, and trend
        long_signal = volume_confirmed and vol_filter and trend_filter and (price_high > h3_12h[i])
        
        # Short conditions: price breaks below L3 level with volume, volatility, and trend
        short_signal = volume_confirmed and vol_filter and trend_filter and (price_low < l3_12h[i])
        
        # Exit when price returns to the opposite side of the pivot level
        pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
        exit_long = position == 1 and price_close < pivot_12h[i]
        exit_short = position == -1 and price_close > pivot_12h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Camarilla breakout with volatility and trend filters for 12h timeframe.
# Enters long when 12h price breaks above weekly H3 level with volume >1.5x average,
# weekly ATR percentile > 0.3 (avoid low volatility), and ADX > 25 (strong trend).
# Enters short when price breaks below weekly L3 level with same conditions.
# Exits when price returns to the weekly pivot level (mean reversion within the week's range).
# Volatility filter prevents trading in choppy markets, trend filter ensures momentum.
# Target: 20-40 trades per year to minimize fee decay while capturing strong weekly trends.