#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Uses 4h Donchian(20) for trend direction to avoid whipsaws in ranging markets.
# Volume > 1.5x 20-period average confirms momentum (moderate threshold to control trade frequency).
# ATR-based stoploss (2.5x) limits drawdown. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag on 1h timeframe.
# Works in bull/bear via 4h Donchian trend filter + volume confirmation + session filter.
# Entry requires 4h Donchian alignment + volume spike + 1h Donchian breakout.

name = "1h_Donchian20_4hTrend_VolumeConfirm_ATRStop_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Calculate ATR(14) for 1h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h Donchian(20) for entry timing
    donchian_high_1h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_1h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(donchian_high_4h_aligned[i]) or
            np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high_4h = donchian_high_4h_aligned[i]
        curr_donchian_low_4h = donchian_low_4h_aligned[i]
        curr_donchian_mid_4h = donchian_mid_4h_aligned[i]
        curr_atr = atr[i]
        curr_donchian_high_1h = donchian_high_1h[i]
        curr_donchian_low_1h = donchian_low_1h[i]
        
        # Volume confirmation: volume > 1.5x 20-period average (moderate threshold to control trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 1h Donchian high, price above 4h Donchian mid, volume spike
            if (curr_close > curr_donchian_high_1h and 
                curr_close > curr_donchian_mid_4h and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: price breaks below 1h Donchian low, price below 4h Donchian mid, volume spike
            elif (curr_close < curr_donchian_low_1h and 
                  curr_close < curr_donchian_mid_4h and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below 1h Donchian low OR stoploss hit
            if (curr_close < curr_donchian_low_1h or 
                curr_close < entry_price - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above 1h Donchian high OR stoploss hit
            if (curr_close > curr_donchian_high_1h or 
                curr_close > entry_price + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals