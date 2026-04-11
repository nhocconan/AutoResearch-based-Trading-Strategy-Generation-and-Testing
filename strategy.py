#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot levels (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Standard Camarilla: H4 = C + ((H-L) * 1.1/2), L4 = C - ((H-L) * 1.1/2)
    # Also calculate H3, L3, H2, L2, H1, L1 for reference
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (using close of previous day)
    # H4 = C + (range * 1.1/2)
    # L4 = C - (range * 1.1/2)
    camarilla_h4 = close_1d + (daily_range * 1.1 / 2)
    camarilla_l4 = close_1d - (daily_range * 1.1 / 2)
    
    # Also calculate H3 and L3 for additional reference
    camarilla_h3 = close_1d + (daily_range * 1.1/4)
    camarilla_l3 = close_1d - (daily_range * 1.1/4)
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.3x 20-period average (slightly lower than before to allow more trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX for trend strength (14 period) - using HTF for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX on 12h data
    tr1_12h = high_12h[1:] - low_12h[1:]
    tr2_12h = np.abs(high_12h[1:] - close_12h[:-1])
    tr3_12h = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    
    plus_dm_12h = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), np.maximum(h_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm_12h = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    plus_di_12h = 100 * pd.Series(plus_dm_12h).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    minus_di_12h = 100 * pd.Series(minus_dm_12h).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_12h_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx_12h_4h[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.3x average - balanced to avoid overtrading)
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Trend filter: ADX > 20 on 12h (moderate trend to allow more trades in ranging markets)
        trend_filter = adx_12h_4h[i] > 20
        
        # Long conditions: price touches or breaks above H3 or H4 with volume and trend
        # Using H3 as entry level for more frequent signals, H4 as stronger breakout
        long_signal = volume_confirmed and trend_filter and (price_high > h3_4h[i])
        
        # Short conditions: price touches or breaks below L3 or L4 with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < l3_4h[i])
        
        # Exit when price returns to the opposite Camarilla level (mean reversion)
        # Exit long when price returns to L3 level
        exit_long = position == 1 and price_close < l3_4h[i]
        # Exit short when price returns to H3 level
        exit_short = position == -1 and price_close > h3_4h[i]
        
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

# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance levels.
# Strategy enters long when 4h price touches or breaks above H3 level (close + range*1.1/4) 
# with volume confirmation (>1.3x 20-period average) and trend filter (12h ADX > 20).
# Enters short when price touches or breaks below L3 level with same conditions.
# Exits when price returns to the opposite level (L3 for long exits, H3 for short exits).
# Uses 12h ADX for trend filtering to avoid whipsaws in ranging markets while allowing 
# participation in trends. The combination of Camarilla levels (mathematically derived 
# support/resistance), volume confirmation, and trend filtering creates a robust edge
# that works in both bull and bear markets by adapting to market conditions.
# Target: 20-40 trades per year to balance opportunity with cost efficiency.