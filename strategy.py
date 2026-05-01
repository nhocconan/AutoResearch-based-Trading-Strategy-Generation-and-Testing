#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA34 trend and volume confirmation.
# Long when: BB width at 20-period low + price breaks above upper BB AND 1d close > 1d EMA34 AND volume > 2.0x 20-period average
# Short when: BB width at 20-period low + price breaks below lower BB AND 1d close < 1d EMA34 AND volume > 2.0x 20-period average
# Uses Bollinger Bands from 4h for volatility contraction/expansion, 1d EMA34 for trend alignment, volume spike for conviction.
# Target: 20-50 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing explosive moves.
# Works in bull (breakouts with trend) and bear (breakouts with trend) by trading with aligned 1d trend.

name = "4h_BollingerSqueeze_1dEMA34_VolumeBreakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Bollinger Bands calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Bollinger Bands (20, 2)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # BB width 20-period low (squeeze condition)
    bb_width_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    squeeze_condition = bb_width <= bb_width_low * 1.1  # Within 10% of 20-period low
    
    # Align Bollinger Bands and squeeze condition to 4h primary timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    squeeze_aligned = align_htf_to_ltf(prices, df_4h, squeeze_condition)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for 20-period BB + 20-period width low + 34 EMA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_upper_bb = upper_bb_aligned[i]
        curr_lower_bb = lower_bb_aligned[i]
        curr_squeeze = squeeze_aligned[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        # Calculate 4h volume MA on the fly using aligned 4h data
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # 1d trend filter
        uptrend_1d = curr_close > curr_ema_34
        downtrend_1d = curr_close < curr_ema_34
        
        # Breakout conditions
        breakout_up = curr_high > curr_upper_bb  # Using high for breakout confirmation
        breakout_down = curr_low < curr_lower_bb  # Using low for breakout confirmation
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze + breakout above upper BB AND 1d uptrend AND volume confirmation
            if (curr_squeeze and breakout_up and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + breakout below lower BB AND 1d downtrend AND volume confirmation
            elif (curr_squeeze and breakout_down and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to middle BB (mean reversion) OR 1d trend turns down
            middle_bb = (curr_upper_bb + curr_lower_bb) / 2
            if (curr_close < middle_bb) or \
               not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to middle BB (mean reversion) OR 1d trend turns up
            middle_bb = (curr_upper_bb + curr_lower_bb) / 2
            if (curr_close > middle_bb) or \
               not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals