#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# - Long: Price breaks above Donchian upper channel (20-period high) + volume > 1.3x 20-period average + 1d ATR ratio < 0.8 (low volatility)
# - Short: Price breaks below Donchian lower channel (20-period low) + volume > 1.3x 20-period average + 1d ATR ratio < 0.8 (low volatility)
# - Exit: ATR-based trailing stop (2.5 ATR from extreme) or opposite Donchian breakout
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - ATR volatility filter ensures breakouts occur in low-volatility environments, reducing false signals
# - Volume confirmation filters out weak breakouts and increases signal quality
# - ATR stoploss manages risk during volatile periods

name = "4h_1d_donchian_breakout_atr_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ATR for volatility filter (20-period ATR)
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_20_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Pre-compute 1d ATR ratio (current ATR / 50-period average) for regime filter
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    atr_ratio_1d = np.where(atr_50_1d_aligned > 0, atr_20_1d_aligned / atr_50_1d_aligned, 1.0)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(60, n):  # Start after 60-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Volatility filter: 1d ATR ratio < 0.8 (low volatility environment)
        vol_filter = atr_ratio_1d[i] < 0.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above upper Donchian channel with volume confirmation and low volatility
        if close_price > upper_channel and vol_confirm and vol_filter:
            enter_long = True
        
        # Short breakout: price closes below lower Donchian channel with volume confirmation and low volatility
        if close_price < lower_channel and vol_confirm and vol_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below lower channel
            exit_long = (close_price <= long_stop) or (close_price < lower_channel)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above upper channel
            exit_short = (close_price >= short_stop) or (close_price > upper_channel)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.5 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.5 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2.5*ATR)
            long_stop = max(long_stop, high[i] - 2.5 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2.5*ATR)
            short_stop = min(short_stop, low[i] + 2.5 * atr_14[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals