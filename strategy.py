#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla pivot calculation (previous week)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Resistance and support levels (previous week's data)
    r3 = close_1w + range_1w * 1.166
    s3 = close_1w - range_1w * 1.166
    
    # Shift by 1 to use only completed weekly bars (previous week's levels)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r3[0] = np.nan
    s3[0] = np.nan
    
    # Align weekly Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1w, r3)
    s3_12h = align_htf_to_ltf(prices, df_1w, s3)
    
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
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
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
        
        # Trend filter: ADX > 20 (balanced to reduce trades while keeping edge)
        trend_filter = adx[i] > 20
        
        # Long conditions: price breaks above R3 with volume and trend
        long_signal = volume_confirmed and trend_filter and (price_high > r3_12h[i])
        
        # Short conditions: price breaks below S3 with volume and trend
        short_signal = volume_confirmed and trend_filter and (price_low < s3_12h[i])
        
        # Exit when price returns to the weekly pivot (mean reversion)
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

# Hypothesis: Weekly Camarilla pivot breakout strategy for 12h timeframe with volume confirmation (>1.5x average volume) and ADX filter (>20).
# Enters long when 12h price breaks above weekly R3 level (close + 1.166*range) with volume >1.5x average and ADX>20.
# Enters short when price breaks below weekly S3 level (close - 1.166*range) with same conditions.
# Exits when price returns to the weekly pivot level (mean reversion within the week's range).
# Uses R3/S3 levels (not R4/S4) to reduce false breakouts and increase win rate.
# Moderate ADX threshold balances trade frequency and signal quality.
# Reduced position size to 0.25 to lower risk and drawdown.
# Target: 15-35 trades per year to minimize fee drift while capturing strong weekly trends.
# Camarilla pivots work well in both bull and bear markets as they adapt to weekly volatility ranges.