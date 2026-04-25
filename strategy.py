#!/usr/bin/env python3
"""
1h_Keltner_Channel_VolumeSpike_4hTrend_Session
Hypothesis: 1h Keltner Channel breakout with 4h EMA50 trend filter and volume confirmation, 
restricted to 08-20 UTC session. Uses higher timeframe (4h) for signal direction to reduce 
overtrading on 1h, while 1h provides precise entry timing. Volume spike confirms momentum. 
Session filter avoids low-liquidity periods. Designed for 15-37 trades/year on BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA50 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h ATR for Keltner Channel (20-period EMA of True Range)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h EMA20 for Keltner Channel middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel upper and lower bands
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    # Volume spike: current volume > 2.0 * 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for indicators (max 50 for 4h EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(vol_ema[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume spike and trend alignment
            # Long breakout: price breaks above Keltner Upper with uptrend and volume spike
            long_breakout = (curr_close > keltner_upper[i]) and uptrend and volume_spike[i]
            # Short breakout: price breaks below Keltner Lower with downtrend and volume spike
            short_breakout = (curr_close < keltner_lower[i]) and downtrend and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below Keltner Lower (mean reversion) or trend changes
            elif curr_close < keltner_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above Keltner Upper (mean reversion) or trend changes
            elif curr_close > keltner_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Keltner_Channel_VolumeSpike_4hTrend_Session"
timeframe = "1h"
leverage = 1.0