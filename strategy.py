#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Daily Donchian channel breakouts capture major trend moves. 
The 1-week EMA34 filter ensures we trade with higher timeframe momentum, 
reducing false breakouts in both bull and bear markets. Volume confirmation 
ensures institutional participation. ATR-based stoploss manages risk. 
Designed for very low trade frequency (7-25/year) to minimize fee drag on 1d timeframe.
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
    
    # Get 1d data for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d high/low
    # Upper = max(high over last 20 days), Lower = min(low over last 20 days)
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate ATR(14) for stoploss and position sizing
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need enough for Donchian, EMA, ATR, volume MA
    start_idx = max(34, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        donchian_high_level = donchian_high_aligned[i]
        donchian_low_level = donchian_low_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > donchian_high_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Donchian low AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < donchian_low_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_at_entry = atr_val
            elif short_entry:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: ATR-based stoploss OR price closes below Donchian low (reversal)
            stop_loss = entry_price - (2.5 * atr_at_entry)
            if (curr_close < stop_loss) or (curr_close < donchian_low_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: ATR-based stoploss OR price closes above Donchian high (reversal)
            stop_loss = entry_price + (2.5 * atr_at_entry)
            if (curr_close > stop_loss) or (curr_close > donchian_high_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0