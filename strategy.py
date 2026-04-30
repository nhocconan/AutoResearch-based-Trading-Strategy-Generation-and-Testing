#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 1w EMA34 for stable multi-week trend direction and requires volume > 2.0x 20-period average.
# Only takes bullish breakouts above upper Donchian channel in uptrend or bearish breakouts below lower Donchian channel in downtrend.
# Added ATR-based stoploss (2.0x ATR) and profit target at opposite Donchian level.
# Designed for very low trade frequency (~10-20 trades/year) to minimize fee drag and avoid overtrading.
# Donchian channels provide reliable swing points that work in both trending and ranging markets.

name = "12h_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for stoploss (using 14-period ATR on 12h)
    if n >= 14:
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        atr = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Donchian channels on 12h timeframe (20-period lookback)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.nan
            donchian_low = np.nan
        
        # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, 1w EMA34 uptrend, volume spike
            if (not np.isnan(donchian_high) and
                curr_close > donchian_high and 
                curr_close > curr_ema_34_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower Donchian, 1w EMA34 downtrend, volume spike
            elif (not np.isnan(donchian_low) and
                  curr_close < donchian_low and 
                  curr_close < curr_ema_34_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below lower Donchian (opposite channel), or ATR stoploss hit
            if (not np.isnan(donchian_low) and curr_close < donchian_low) or \
               curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above upper Donchian (opposite channel), or ATR stoploss hit
            if (not np.isnan(donchian_high) and curr_close > donchian_high) or \
               curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals