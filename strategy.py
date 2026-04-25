#!/usr/bin/env python3
"""
1h_HTF_Trend_LTF_Pullback_v1
Hypothesis: On 1h timeframe, trade pullbacks to EMA20 in direction of 4h EMA50 trend during London/NY session (08-20 UTC) with volume confirmation (>1.2x average). Uses discrete sizing (0.20) to minimize fee churn. Designed for 15-35 trades/year per symbol by requiring HTF trend alignment + LTF pullback + volume spike + session filter. Works in bull/bear by following HTF trend and using mean-reversion entries within trend.
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
    
    # Precompute session filter (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate EMA(20) on 1h for LTF entries - vectorized
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # 4h EMA50 needs 50, 1h EMA20 needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_20_1h[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get values
        ema_50_val = ema_50_4h_aligned[i]
        ema_20_val = ema_20_1h[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume spike condition: current volume > 1.2x 20-period average
        volume_spike = vol_val > 1.2 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: pullback to EMA20 in direction of 4h EMA50 trend
            # Long: price pulls back to or slightly below EMA20 but closes above it, 
            #       while 4h EMA50 is rising (trend up), with volume spike
            # Short: price pulls back to or slightly above EMA20 but closes below it,
            #        while 4h EMA50 is falling (trend down), with volume spike
            
            # 4h EMA50 trend direction (using previous bar to avoid look-ahead)
            ema_50_prev = ema_50_4h_aligned[i-1] if i > 0 else ema_50_val
            ema_50_rising = ema_50_val > ema_50_prev
            ema_50_falling = ema_50_val < ema_50_prev
            
            # Price pulled back to EMA20 (within 0.5%) and closed in direction of trend
            pullback_long = (low_val <= ema_20_val * 1.005) and (close_val > ema_20_val)
            pullback_short = (high_val >= ema_20_val * 0.995) and (close_val < ema_20_val)
            
            long_signal = pullback_long and ema_50_rising and volume_spike
            short_signal = pullback_short and ema_50_falling and volume_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. Stoploss: 2.5% adverse move
            if close_val < entry_price * 0.975:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Take profit: 5% favorable move
            elif close_val >= entry_price * 1.05:
                signals[i] = 0.0  # flat
                position = 0
                entry_price = 0.0
            # 3. Trend reversal: 4h EMA50 starts falling
            elif i > 0 and ema_50_val < ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. Stoploss: 2.5% adverse move
            if close_val > entry_price * 1.025:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Take profit: 5% favorable move
            elif close_val <= entry_price * 0.95:
                signals[i] = 0.0  # flat
                position = 0
                entry_price = 0.0
            # 3. Trend reversal: 4h EMA50 starts rising
            elif i > 0 and ema_50_val > ema_50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1h_HTF_Trend_LTF_Pullback_v1"
timeframe = "1h"
leverage = 1.0