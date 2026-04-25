#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA34 Trend Filter and Volume Spike
Hypothesis: Donchian breakouts capture strong momentum moves. Using 1w EMA34 as higher-timeframe trend filter ensures alignment with weekly trend, reducing false signals in choppy markets. Volume spike confirms breakout strength. Works in bull markets (breakouts above upper channel) and bear markets (breakdowns below lower channel) by requiring trend alignment. Discrete sizing (0.0, ±0.30) minimizes fee churn. Target: 15-25 trades/year on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for volume spike threshold and stoploss reference
    tr1 = pd.Series(high).rolling(window=1, min_periods=1).max() - pd.Series(low).rolling(window=1, min_periods=1).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike: volume > 2.0 * 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and ATR calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_spike = vol_spike[i]
        ema_trend = ema_34_1w_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_high > donch_high[i]) and curr_vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian channel AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_low < donch_low[i]) and curr_vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below lower Donchian channel OR price < 1w EMA34 (trend change)
            if (curr_low < donch_low[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian channel OR price > 1w EMA34 (trend change)
            if (curr_high > donch_high[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0