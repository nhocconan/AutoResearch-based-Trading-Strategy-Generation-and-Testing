# Your turn. Make it count. Remember the lessons from 16,000+ experiments.
#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian breakout with 1-day ADX trend filter and volume confirmation.
The Donchian channel provides clear breakout levels while the ADX filter ensures
trades only occur in trending markets, avoiding whipsaws in ranging conditions.
Volume spikes confirm institutional participation. This approach works in both bull
and bear markets by capturing strong directional moves when volatility expands.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12-period Donchian channel on 12h data
    donchian_period = 12
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 12-period volume average for spike detection
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_avg_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper channel, trending market, volume spike
            if (close[i] > upper_channel[i] and
                is_trending and
                volume[i] > 2.0 * vol_avg_12[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel, trending market, volume spike
            elif (close[i] < lower_channel[i] and
                  is_trending and
                  volume[i] > 2.0 * vol_avg_12[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite channel or ADX drops below 20 (range)
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower channel or ADX weakens
                if close[i] < lower_channel[i] or adx_1d_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper channel or ADX weakens
                if close[i] > upper_channel[i] or adx_1d_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0