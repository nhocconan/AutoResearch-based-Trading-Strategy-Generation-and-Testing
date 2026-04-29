#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume spike + ATR(14) stoploss
# Donchian channels identify structural breakouts; 1d EMA50 filters for higher timeframe trend;
# volume confirms breakout strength; ATR stoploss manages risk. Works in bull/bear via trend filter.
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag on 12h timeframe.

name = "12h_Donchian20_Breakout_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) from previous candles
    # Upper = max(high of last 20 periods), Lower = min(low of last 20 periods)
    # We use shifted data to avoid look-ahead: donchian_upper[i] = max(high[i-20:i])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(100, 50, 20, 14)  # warmup for Donchian, EMA50, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_price = entry_price - 2.0 * atr_at_entry
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Price crosses below 1d EMA50 (trend change)
            # 3. Price drops below Donchian lower (breakout failed)
            if (curr_low <= stop_price or
                curr_close < curr_ema_50_1d or
                curr_close < curr_lower):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_price = entry_price + 2.0 * atr_at_entry
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Price crosses above 1d EMA50 (trend change)
            # 3. Price rises above Donchian upper (breakout failed)
            if (curr_high >= stop_price or
                curr_close > curr_ema_50_1d or
                curr_close > curr_upper):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + above 1d EMA50 + volume confirm
            if (curr_close > curr_upper and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: price breaks below Donchian lower + below 1d EMA50 + volume confirm
            elif (curr_close < curr_lower and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals