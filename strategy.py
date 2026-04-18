#!/usr/bin/env python3
"""
1h_LiquidityZone_4hTrend_1dVolatilityFilter
Hypothesis: 1-hour liquidity zones (equal highs/lows) act as support/resistance. 
Trades are taken in direction of 4-hour trend (EMA20) with 1-day volatility filter to avoid chop.
Volume confirmation on breakout ensures institutional participation. 
Designed for 1h timeframe with strict entry to limit trades to 15-35/year.
Works in bull/bear via trend filter and volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close']
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volatility filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    atr_period = 14
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    volatility_filter = atr_1d_aligned > atr_ma_1d_aligned  # Only trade when volatility is expanding
    
    # Precompute 1h equal highs/lows (liquidity zones) - look back 20 bars
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    lookback = 20
    tolerance = 0.001  # 0.1% tolerance for equal levels
    
    for i in range(lookback, n):
        # Check for equal highs (within tolerance)
        high_window = high[i-lookback:i]
        max_high = np.max(high_window)
        if high[i] <= max_high * (1 + tolerance) and high[i] >= max_high * (1 - tolerance):
            equal_high[i] = True
        
        # Check for equal lows (within tolerance)
        low_window = low[i-lookback:i]
        min_low = np.min(low_window)
        if low[i] <= min_low * (1 + tolerance) and low[i] >= min_low * (1 - tolerance):
            equal_low[i] = True
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_20_4h_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above equal high liquidity zone, uptrend, volatility expanding, volume spike
            if equal_high[i] and price > ema_trend and vol_filter and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below equal low liquidity zone, downtrend, volatility expanding, volume spike
            elif equal_low[i] and price < ema_trend and vol_filter and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position: exit on breakdown of equal low or trend reversal
            signals[i] = 0.20
            if equal_low[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: exit on breakout of equal high or trend reversal
            signals[i] = -0.20
            if equal_high[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_LiquidityZone_4hTrend_1dVolatilityFilter"
timeframe = "1h"
leverage = 1.0