#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_trend_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily OHLC for Keltner channel calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(20) and ATR(10) for Keltner channel
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema_20 + 2 * atr_10
    keltner_lower = ema_20 - 2 * atr_10
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    keltner_upper = np.roll(keltner_upper, 1)
    keltner_lower = np.roll(keltner_lower, 1)
    keltner_upper[0] = np.nan
    keltner_lower[0] = np.nan
    
    # Align daily Keltner levels to 4h timeframe
    upper_4h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    lower_4h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    
    # 4h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ADX for trend strength (14 period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = tr[1:]
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
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
        
        # Long conditions: price breaks above upper Keltner with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > upper_4h[i])
        
        # Short conditions: price breaks below lower Keltner with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < lower_4h[i])
        
        # Exit when price returns to the opposite side of the EMA (mean reversion)
        ema_4h = align_htf_to_ltf(prices, df_1d, ema_20)
        exit_long = position == 1 and price_close < ema_4h[i]
        exit_short = position == -1 and price_close > ema_4h[i]
        
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

# Hypothesis: Daily Keltner breakout strategy for 4h timeframe with ADX filter (>25) and volume confirmation (>1.5x average volume).
# Enters long when 4h price breaks above daily upper Keltner band (EMA20 + 2*ATR10) with volume >1.5x average and ADX>25.
# Enters short when price breaks below daily lower Keltner band (EMA20 - 2*ATR10) with same conditions.
# Exits when price returns to the daily EMA20 (mean reversion within the day's range).
# Higher ADX threshold reduces trade frequency to avoid overtrading while maintaining edge in strong trends.
# Target: 20-30 trades per year to minimize fee drift while capturing strong daily trends.
# This strategy targets a less saturated approach (Keltner channels) compared to the overused Camarilla and Donchian channels,
# while still capturing the core principle of price channel breakouts with volume and trend confirmation.
# Keltner channels adapt better to volatility changes than fixed percentage channels, making them suitable for both bull and bear markets.
# The 4h timeframe provides a balance between capturing daily moves and reducing noise compared to lower timeframes.