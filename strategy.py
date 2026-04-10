#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
# - Long: price > Donchian high(20) AND 1d close > EMA(50) AND volume > 1.5x 20-period average
# - Short: price < Donchian low(20) AND 1d close < EMA(50) AND volume > 1.5x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 4h timeframe: targets 19-50 trades/year to avoid fee drag
# - Works in bull/bear markets: EMA filter ensures we trade with the higher timeframe trend, avoiding chop

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
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
            # Exit: price < Donchian low OR stoploss hit
            if close_4h[i] < donchian_low[i] or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR stoploss hit
            if close_4h[i] > donchian_high[i] or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with EMA trend and volume filters
            if vol_spike[i]:
                # Long: price > Donchian high(20) AND 1d close > EMA(50)
                if close_4h[i] > donchian_high[i] and close_4h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price < Donchian low(20) AND 1d close < EMA(50)
                elif close_4h[i] < donchian_low[i] and close_4h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals