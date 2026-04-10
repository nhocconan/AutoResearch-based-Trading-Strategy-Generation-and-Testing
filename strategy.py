#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter + volume confirmation
# - Primary signal: Price breaks above/below 20-period Donchian channel on 12h
# - Trend filter: 1w close > EMA(50) for longs, < EMA(50) for shorts (institutional trend)
# - Volume filter: 12h volume > 1.3x 20-period average volume (momentum confirmation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts; 1w trend filter avoids counter-trend trades

name = "12h_1w_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Pre-compute 12h Donchian Channel (20)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Upper band: highest high of last 20 periods
    upper_donchian = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_donchian = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Pre-compute 12h volume spike filter
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.3 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below middle band OR stoploss hit
            if close_12h[i] < middle_donchian[i] or close_12h[i] < entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band OR stoploss hit
            if close_12h[i] > middle_donchian[i] or close_12h[i] > entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above upper Donchian in uptrend (close > EMA50)
                if close_12h[i] > upper_donchian[i] and close_12h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower Donchian in downtrend (close < EMA50)
                elif close_12h[i] < lower_donchian[i] and close_12h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals