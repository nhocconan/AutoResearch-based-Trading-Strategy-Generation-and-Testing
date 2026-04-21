#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeRegime_ATRStop
Hypothesis: 4h Donchian(20) breakout aligned with 12h EMA34 trend and volume spike filter.
Enter long on upper band break when 12h EMA34 trending up and volume > 1.5x 20-period MA.
Enter short on lower band break when 12h EMA34 trending down and volume > 1.5x 20-period MA.
Exit on opposite band break or ATR(14) trailing stop (2.0*ATR).
Designed for moderate trade frequency (target: 25-40 trades/year) to balance edge and fees.
Works in bull/bear via 12h trend alignment and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for HTF trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: upper = max(high, lookback=20), lower = min(low, lookback=20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_upper[i]) 
            or np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long conditions: price > upper Donchian, 12h EMA34 uptrend, volume spike
            long_breakout = price > donchian_upper[i]
            long_trend = price > ema_34_12h_aligned[i]
            
            # Short conditions: price < lower Donchian, 12h EMA34 downtrend, volume spike
            short_breakout = price < donchian_lower[i]
            short_trend = price < ema_34_12h_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below lower Donchian (support broken)
            elif price < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above upper Donchian (resistance broken)
            elif price > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeRegime_ATRStop"
timeframe = "4h"
leverage = 1.0