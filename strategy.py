#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_VolumeFilter_v2
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d trend regime and volume confirmation.
- Bullish regime: 1d close > 1d EMA50 → look for Bull Power expansion + volume spike for longs
- Bearish regime: 1d close < 1d EMA50 → look for Bear Power expansion + volume spike for shorts
- Uses 6h timeframe for entries, 1d for trend filter (proven combo from prior experiments)
- Volume filter (>1.8x 20-period average) reduces false signals
- Discrete position sizing: 0.25 (25% of capital) to manage drawdown
- Target: 12-30 trades/year per symbol (50-120 total over 4 years) to minimize fee drag
- Works in bull/bear via 1d trend alignment as regime filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 6h OHLC for Elder Ray calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA13 for Elder Ray (on 6h close)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine 1d trend regime: bullish if close > EMA50, bearish if close < EMA50
    # We need the 1d close price aligned to 6h bars
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    regime_bullish = close_1d_aligned > ema_50_1d_aligned  # 1d uptrend regime
    regime_bearish = close_1d_aligned < ema_50_1d_aligned  # 1d downtrend regime
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.8x 20-period average
            vol_filter = volume[i] > 1.8 * vol_ma[i]
            
            # Long conditions: 1d bullish regime + Bull Power expansion + volume filter
            # Bull Power expansion: current Bull Power > previous Bull Power (momentum building)
            bull_power_expansion = bull_power[i] > bull_power[i-1] if i > 0 else False
            
            # Short conditions: 1d bearish regime + Bear Power expansion + volume filter
            # Bear Power expansion: current Bear Power > previous Bear Power (momentum building)
            bear_power_expansion = bear_power[i] > bear_power[i-1] if i > 0 else False
            
            # Entry logic
            if regime_bullish[i] and bull_power_expansion and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif regime_bearish[i] and bear_power_expansion and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if Bull Power contracts (loss of bullish momentum)
            elif bull_power[i] < bull_power[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if Bear Power contracts (loss of bearish momentum)
            elif bear_power[i] < bear_power[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_VolumeFilter_v2"
timeframe = "6h"
leverage = 1.0