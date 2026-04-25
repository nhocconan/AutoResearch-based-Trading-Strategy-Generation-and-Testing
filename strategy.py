#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
The 1-week EMA34 filter ensures we trade with the primary trend, reducing whipsaws in ranging markets.
Volume spike confirms institutional participation. Designed for low trade frequency (7-25/year) 
on daily timeframe to minimize fee drag. Works in both bull and bear markets: 
- In bull markets: breakouts above upper band with uptrend filter capture rallies
- In bear markets: breakouts below lower band with downtrend filter capture crashes
- Volume confirmation avoids false breakouts
- ATR-based stoploss manages risk
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    # Use rolling window on daily high/low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period ATR for stoploss and volume context
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, EMA, ATR, volume MA
    start_idx = max(20, 34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian band AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > upper_band) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower Donchian band AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < lower_band) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below lower Donchian band (reversal) OR 2*ATR stoploss
            if (curr_close < lower_band) or (curr_close < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above upper Donchian band (reversal) OR 2*ATR stoploss
            if (curr_close > upper_band) or (curr_close > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0