#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v4"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for the previous day (to avoid look-ahead)
    camarilla_high = (close_1d + (high_1d - low_1d) * 1.1 / 6)
    camarilla_low = (close_1d - (high_1d - low_1d) * 1.1 / 6)
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan
    camarilla_low[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_high_4h = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_4h = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
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
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_4h[i]) or np.isnan(camarilla_low_4h[i]) or 
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
        
        # Long conditions: price breaks above upper Camarilla level with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > camarilla_high_4h[i])
        
        # Short conditions: price breaks below lower Camarilla level with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < camarilla_low_4h[i])
        
        # Exit when price returns to the midpoint of the Camarilla range (mean reversion)
        camarilla_mid = (camarilla_high_4h[i] + camarilla_low_4h[i]) / 2
        exit_long = position == 1 and price_close < camarilla_mid
        exit_short = position == -1 and price_close > camarilla_mid
        
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

# Hypothesis: Camarilla breakout strategy for 4h timeframe with ADX filter (>25) and volume confirmation (>1.5x average volume).
# Enters long when 4h price breaks above daily upper Camarilla level (close + 1.1*(H-L)/6) with volume >1.5x average and ADX>25.
# Enters short when price breaks below daily lower Camarilla level (close - 1.1*(H-L)/6) with same conditions.
# Exits when price returns to the midpoint of the Camarilla range (mean reversion within the day's range).
# Higher ADX threshold reduces trade frequency to avoid overtrading while maintaining edge in strong trends.
# Target: 20-30 trades per year to minimize fee drift while capturing strong daily trends.
# Camarilla levels provide dynamic support/resistance based on previous day's volatility, adapting to both bull and bear markets.
# The 4h timeframe provides a balance between capturing daily moves and reducing noise compared to lower timeframes.
# Daily timeframe is used to capture longer-term trends and avoid noise from shorter-term fluctuations.