#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h trend filter (EMA50) + volume confirmation
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 12h EMA50 (uptrend) AND volume > 1.5x 20-period average
# - Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 12h EMA50 (downtrend) AND volume > 1.5x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: trend filter prevents counter-trend trades, Elder Ray captures momentum shifts

name = "6h_12h_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Pre-compute 6h Elder Ray components
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low_6h   # Bear Power = EMA13 - Low
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: momentum shifts bearish OR stoploss hit
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: momentum shifts bullish OR stoploss hit
            if bear_power[i] <= 0 or bull_power[i] >= 0 or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with trend and volume filters
            if vol_spike[i]:
                # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 12h EMA50 (uptrend)
                if bull_power[i] > 0 and bear_power[i] < 0 and close_6h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 12h EMA50 (downtrend)
                elif bear_power[i] > 0 and bull_power[i] < 0 and close_6h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals