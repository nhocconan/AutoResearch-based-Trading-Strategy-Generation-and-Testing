#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_volume_trend_v1"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channel (20 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed weekly bars (previous week's levels)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # Align weekly Donchian levels to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1w, donchian_high)
    lower_12h = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # 12h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX for trend strength (14 period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = tr[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 (strong trend filter to reduce trades)
        trend_filter = adx[i] > 25
        
        # Long conditions: price breaks above upper Donchian with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > upper_12h[i])
        
        # Short conditions: price breaks below lower Donchian with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < lower_12h[i])
        
        # Exit when price returns to the opposite side of the Donchian midpoint (mean reversion)
        donchian_mid = (upper_12h + lower_12h) / 2
        exit_long = position == 1 and price_close < donchian_mid[i]
        exit_short = position == -1 and price_close > donchian_mid[i]
        
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

# Hypothesis: Weekly Donchian breakout strategy for 12h timeframe with ADX filter (>25) and volume confirmation (>1.5x average volume).
# Enters long when 12h price breaks above weekly upper Donchian band (20-period high) with volume >1.5x average and ADX>25.
# Enters short when price breaks below weekly lower Donchian band (20-period low) with same conditions.
# Exits when price returns to the weekly Donchian midpoint (mean reversion within the week's range).
# Higher ADX threshold reduces trade frequency to avoid overtrading while maintaining edge in strong trends.
# Target: 15-25 trades per year to minimize fee drift while capturing strong weekly trends.
# Donchian channels provide clear support/resistance levels that work in both trending and ranging markets.
# The 12h timeframe provides a balance between capturing weekly moves and reducing noise compared to lower timeframes.
# Weekly timeframe is used to capture longer-term trends and avoid noise from shorter-term fluctuations.