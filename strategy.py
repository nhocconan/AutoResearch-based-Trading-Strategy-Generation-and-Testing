#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel (20-period high) in bull trend (close > 1d EMA34) with volume > 2.0x 20-period MA.
# Short when price breaks below Donchian lower channel (20-period low) in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. ATR-based stoploss exits when price moves against position by 2.5x ATR(20).
# Donchian channels provide clear structure, EMA34 filters for higher-timeframe trend, volume confirms institutional participation.
# Target: 75-150 total trades over 4 years (19-38/year). Works in both bull and bear markets via trend filter.

name = "4h_Donchian20_1dEMA34_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(20) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Donchian breakout conditions
        breakout_up = close_val > upper  # break above upper channel
        breakout_down = close_val < lower  # break below lower channel
        
        # Update trailing stop for existing positions
        if position == 1:
            # Long stoploss: price drops below entry_price - 2.5 * atr
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short stoploss: price rises above entry_price + 2.5 * atr
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:  # position == 0, look for new entries
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
    
    return signals