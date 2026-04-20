#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily CCI(20) mean reversion with weekly trend filter and volume confirmation
# In ranging markets: long when CCI < -100, short when CCI > +100
# Weekly trend filter: only trade long when price > weekly EMA20, short when price < weekly EMA20
# Volume filter requires above-average participation to avoid false signals
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 (trend filter)
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Load daily data ONCE for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily CCI(20)
    tp = (high_1d + low_1d + close_1d) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - ma_tp) / (0.015 * mad)
    
    # Align daily CCI to 1d timeframe (no alignment needed as we're using 1d timeframe)
    # But we still need to align the weekly EMA to daily
    
    # Calculate daily ATR for stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: daily volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(cci[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        price = close[i]
        weekly_ema = ema20_1w_aligned[i]
        cci_value = cci[i]
        
        if position == 0:
            # Long: CCI oversold (< -100), price above weekly EMA (uptrend), with volume
            if cci_value < -100 and price > weekly_ema and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: CCI overbought (> +100), price below weekly EMA (downtrend), with volume
            elif cci_value > 100 and price < weekly_ema and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or CCI returns to neutral (> -50)
            if price <= entry_price - 2.0 * atr[i] or cci_value > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or CCI returns to neutral (< +50)
            if price >= entry_price + 2.0 * atr[i] or cci_value < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_CCI20_WeeklyEMA20_VolumeFilter"
timeframe = "1d"
leverage = 1.0