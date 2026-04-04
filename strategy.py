#!/usr/bin/env python3
"""
exp_6462_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA direction filter and volume confirmation.
- 1d EMA(50) determines bias: long when close > EMA50, short when close < EMA50.
- Donchian breakout provides entry timing in direction of 1d bias.
- Volume confirmation (volume > 1.5x 20-period average) ensures momentum behind breakout.
- Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets.
- Target: 50-150 trades over 4 years (12-37/year) with discrete sizing to minimize fees.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6462_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels on 12h data
    lookback = 20
    donchian_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().shift(1)
    donchian_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().shift(1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean()
    volume_ok = prices['volume'] > (1.5 * vol_ma)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    
    start_idx = max(lookback, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if volume confirmation not met
        if not volume_ok.iloc[i]:
            # If in position, check stoploss
            if position_side == 1 and prices['close'].iloc[i] < entry_price - 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            elif position_side == -1 and prices['close'].iloc[i] > entry_price + 2.5 * atr_14[i]:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = signals[i-1]  # hold current signal
            continue
        
        # Get current values
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema = ema_50_aligned[i]
        dh = donchian_high.iloc[i]
        dl = donchian_low.iloc[i]
        
        # Long condition: price breaks above Donchian high AND above 1d EMA50
        if position_side != 1:  # not already long
            if price_high > dh and price_close > ema:
                signals[i] = 0.30  # 30% long
                position_side = 1
                entry_price = price_close
            # Short condition: price breaks below Donchian low AND below 1d EMA50
            elif price_low < dl and price_close < ema:
                signals[i] = -0.30  # 30% short
                position_side = -1
                entry_price = price_close
            else:
                # Hold current signal or flatten based on stoploss
                if position_side == 1:
                    if price_close < entry_price - 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                elif position_side == -1:
                    if price_close > entry_price + 2.5 * atr_14[i]:
                        signals[i] = 0.0
                        position_side = 0
                    else:
                        signals[i] = signals[i-1]
                else:
                    signals[i] = 0.0
        else:
            # Already in position, manage stoploss
            if position_side == 1:
                if price_close < entry_price - 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
            elif position_side == -1:
                if price_close > entry_price + 2.5 * atr_14[i]:
                    signals[i] = 0.0
                    position_side = 0
                else:
                    signals[i] = signals[i-1]
    
    return signals