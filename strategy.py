#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation + ATR stoploss
# - Donchian breakout: price > highest(high,20) for long, < lowest(low,20) for short
# - Trend filter: price > 12h EMA50 for long bias, < 12h EMA50 for short bias
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - ATR stoploss: exit when price moves against position by 2.0x ATR(14)
# - Discrete position sizing (0.25) to minimize fee churn
# - Designed for 4h timeframe: targets 19-50 trades/year to avoid fee drag
# - Works in bull/bear markets: trend filter prevents counter-trend trades, Donchian captures breakouts

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
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
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low_4h - np.roll(prices['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if prices['close'].values[i] < donchian_low[i] or prices['close'].values[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if prices['close'].values[i] > donchian_high[i] or prices['close'].values[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price > Donchian high AND price > 12h EMA50 (uptrend)
                if prices['close'].values[i] > donchian_high[i] and prices['close'].values[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = prices['close'].values[i]
                    signals[i] = 0.25
                # Short: price < Donchian low AND price < 12h EMA50 (downtrend)
                elif prices['close'].values[i] < donchian_low[i] and prices['close'].values[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = prices['close'].values[i]
                    signals[i] = -0.25
    
    return signals