#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for primary trend alignment to reduce whipsaws and capture major moves.
# Volume > 1.8x 24-period average confirms strong momentum (balanced threshold for 12h).
# ATR-based stoploss (2.5x) manages risk during volatile periods.
# Designed for low trade frequency (~12-25 trades/year) to minimize fee drag on 12h timeframe.
# Works in bull/bear via 1d EMA34 trend filter + volume confirmation + Donchian structure.
# Entry requires 1d EMA34 alignment + volume spike + Donchian breakout.

name = "12h_Donchian20_1dEMA34_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for 12h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: volume > 1.8x 24-period average (balanced for 12h)
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
            volume_confirm = volume[i] > (1.8 * vol_ma_24)
        else:
            volume_confirm = False
        
        # Calculate Donchian channels (20-period) for breakout
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = curr_high
            donchian_low = curr_low
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, price above 1d EMA34, volume spike
            if (curr_high > donchian_high and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low, price below 1d EMA34, volume spike
            elif (curr_low < donchian_low and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low OR stoploss hit
            if (curr_low < donchian_low or 
                curr_close < entry_price - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high OR stoploss hit
            if (curr_high > donchian_high or 
                curr_close > entry_price + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals