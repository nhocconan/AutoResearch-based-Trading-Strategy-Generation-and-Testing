#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h EMA34 for trend direction
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Daily volume average (20-period) for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR (14-period) for exit
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - prices['close'][:-1])
    tr3 = np.abs(low[1:] - prices['close'][:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned values
        ema_trend = ema_34_12h_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_high = prices['high'].iloc[i]
        current_low = prices['low'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(vol_avg) or np.isnan(current_atr) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8x daily average volume
        vol_spike = current_volume > 1.8 * vol_avg
        
        # Donchian breakout conditions
        upper_break = current_high > donchian_high[i]
        lower_break = current_low < donchian_low[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with 12h uptrend and volume spike
            if upper_break and current_close > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below Donchian lower with 12h downtrend and volume spike
            elif lower_break and current_close < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or ATR stop loss
            if current_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or ATR stop loss
            if current_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals