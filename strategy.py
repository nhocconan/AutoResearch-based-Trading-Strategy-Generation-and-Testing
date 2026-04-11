#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_extreme_volume_rsi"
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
    
    # Daily OHLC for CCI calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily CCI(20)
    tp_1d = (high_1d + low_1d + close_1d) / 3
    sma_tp_20 = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_20 = (tp_1d - sma_tp_20) / (0.015 * mad)
    
    # Align daily CCI to 4h timeframe
    cci_20_4h = align_htf_to_ltf(prices, df_1d, cci_20)
    
    # 4h RSI(14) for mean reversion
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ADX for trend strength (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
        if (np.isnan(cci_20_4h[i]) or np.isnan(rsi_14[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 (strong trend filter to reduce trades)
        trend_filter = adx[i] > 25
        
        # Long conditions: CCI > 100 (overbought) with volume and trend -> short reversal
        short_signal = volume_confirmed and trend_filter and (cci_20_4h[i] > 100)
        
        # Short conditions: CCI < -100 (oversold) with volume and trend -> long reversal
        long_signal = volume_confirmed and trend_filter and (cci_20_4h[i] < -100)
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and (rsi_14[i] >= 40 and rsi_14[i] <= 60)
        exit_short = position == -1 and (rsi_14[i] >= 40 and rsi_14[i] <= 60)
        
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

# Hypothesis: Daily CCI extreme levels with volume confirmation and ADX filter for 4h timeframe.
# Enters long when daily CCI < -100 (oversold) with volume >1.5x average and ADX>25.
# Enters short when daily CCI > 100 (overbought) with volume >1.5x average and ADX>25.
# Exits when 4h RSI returns to neutral zone (40-60).
# CCI extremes identify potential reversal points, volume confirms institutional interest,
# and ADX>25 ensures we only trade in trending markets where reversals are more likely.
# This strategy targets reversals in trending markets, which works in both bull and bear markets.
# Low expected trade frequency due to strict CCI thresholds and ADX filter.