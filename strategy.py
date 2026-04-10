#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# - Donchian(20) breakout captures strong momentum moves in both bull and bear markets
# - 12h EMA(50) filter ensures we trade only in alignment with higher timeframe trend
# - Volume confirmation (current 4h volume > 2.0x 20-period average) filters weak breakouts
# - Designed for 4h timeframe: targets 15-30 trades/year (60-120 total over 4 years) to avoid fee drag
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Works in bull/bear markets: EMA filter adapts to trend direction

name = "4h_12h_donchian_ema_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Pre-compute 4h Donchian(20) channels
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Upper channel: highest high of last 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low_4h - np.roll(prices['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price closes below Donchian lower channel
            if prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or prices['close'].iloc[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price closes above Donchian upper channel
            if prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or prices['close'].iloc[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above Donchian upper channel AND price > 12h EMA(50)
                if prices['close'].iloc[i] > donchian_high[i] and prices['close'].iloc[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Breakout short: price closes below Donchian lower channel AND price < 12h EMA(50)
                elif prices['close'].iloc[i] < donchian_low[i] and prices['close'].iloc[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals