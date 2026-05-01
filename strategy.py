#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w ADX regime filter.
# Long when price breaks above Donchian upper band with volume > 1.5x 20-bar average and 1w ADX < 25 (range).
# Short when price breaks below Donchian lower band with volume confirmation and 1w ADX < 25.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian channels provide clear structure, volume filters low-momentum breaks, ADX ensures mean-reversion edge in low-volatility ranging markets.
# Works in bull (breakouts with volume) and bear (mean reversion in low ADX chop) regimes.

name = "12h_Donchian_20_Breakout_1dVolume_1wADX_v1"
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
    
    # Pre-compute session hours for consistency (12h timeframe less sensitive but keep structure)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ADX(14) for 1w regime filter: <25 = range/low trend (favor mean reversion), >25 = trending
    def true_range(h, l, c):
        # Vectorized TR calculation
        h_l = h - l
        h_pc = np.abs(np.subtract(h, np.roll(c, 1)))
        l_pc = np.abs(np.subtract(l, np.roll(c, 1)))
        h_pc[0] = np.abs(h[0] - c[0])
        l_pc[0] = np.abs(l[0] - c[0])
        return np.maximum(h_l, np.maximum(h_pc, l_pc))
    
    # Load 1w data ONCE before loop for ADX regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate +DM, -DM, TR for ADX
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_1w = true_range(high_1w, low_1w, close_1w)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1w_smooth = wilders_smoothing(tr_1w, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w_smooth
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w_smooth
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian channels (20-bar) for breakout signals
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for all indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Session filter: optional for 12h, but keep for consistency
        # if not (0 <= hours[i] <= 23):  # always true, kept for structure
        #     signals[i] = 0.0
        #     continue
        
        if (np.isnan(atr[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average (balanced for trade frequency)
        vol_ma = volume_ma_20[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_high > highest_high_20[i]  # break above upper band
        breakout_down = curr_low < lowest_low_20[i]   # break below lower band
        
        # ADX regime filter: only trade in low ADX (range/chop) market
        regime_filter = adx_1w_aligned[i] < 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND low ADX regime
            if (breakout_up and 
                volume_confirm and 
                regime_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume confirmation AND low ADX regime
            elif (breakout_down and 
                  volume_confirm and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR ADX rises above 25 (trending starts)
            elif (curr_low >= lowest_low_20[i] and curr_low <= highest_high_20[i]) or \
                 adx_1w_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR ADX rises above 25 (trending starts)
            elif (curr_high >= lowest_low_20[i] and curr_high <= highest_high_20[i]) or \
                 adx_1w_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals