#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Uses 1d ATR(14) to filter for low volatility breakouts (avoiding choppy markets) and requires volume > 2.0x average.
# Designed for low trade frequency (~50-80 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by only taking breakouts in the direction of the 1d EMA50 trend.

name = "4h_Donchian20_1dATR14_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d['atr'] = np.maximum(
        df_1d['high'] - df_1d['low'],
        np.maximum(
            np.abs(df_1d['high'] - df_1d['close'].shift(1)),
            np.abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )
    atr_14_1d = pd.Series(df_1d['atr'].values).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA50 and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr_14_1d = atr_14_1d_aligned[i]
        
        # Calculate Donchian(20) levels using previous 1d bar (completed)
        if len(df_1d) >= 20:
            # Calculate Donchian levels for each 1d bar
            donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
            donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
            
            # Align to 4h timeframe with proper delay (wait for 1d bar to close)
            donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
            donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
        else:
            donchian_high_aligned = np.full(n, np.nan)
            donchian_low_aligned = np.full(n, np.nan)
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Volatility filter: only trade when ATR is below its 50-period MA (low volatility environment)
        if i >= 50:
            atr_ma_50 = np.mean(atr_14_1d_aligned[i-50:i])
            low_volatility = curr_atr_14_1d < atr_ma_50
        else:
            low_volatility = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, 1d EMA50 uptrend, volume spike, low volatility
            if (curr_close > donchian_high_aligned[i] and 
                curr_close > curr_ema_50_1d and 
                volume_confirm and 
                low_volatility):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low, 1d EMA50 downtrend, volume spike, low volatility
            elif (curr_close < donchian_low_aligned[i] and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm and 
                  low_volatility):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low or reverses below entry
            if curr_close < donchian_low_aligned[i] or curr_close < entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high or reverses above entry
            if curr_close > donchian_high_aligned[i] or curr_close > entry_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals