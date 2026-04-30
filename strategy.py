#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume confirmation.
# Uses ATR-based trailing stop (1.5x) for risk management. Designed for low trade frequency (~15-37/year) to minimize fee drag.
# 4h EMA50 provides medium-term trend direction, 1d volume ensures institutional participation,
# 1h Camarilla levels give precise entry timing. Works in bull/bear via trend following.
# Session filter (08-20 UTC) reduces noise trades.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_1dVolumeSpike_ATRTrail_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoid datetime arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA20 for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h Camarilla pivot levels (R1, S1) from previous 1h bar
    # We need to calculate these for each 1h bar using the prior bar's OHLC
    # Shift to get previous bar's high, low, close
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    
    # Calculate 1h ATR(14) for dynamic trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: price above/below 4h EMA50 determines trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_r1 = r1[i]
        curr_s1 = s1[i]
        curr_vol_ma_20 = vol_ma_20_aligned[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        volume_spike = curr_volume > (1.5 * curr_vol_ma_20)
        
        if position == 0:  # Flat - look for new entries
            if is_uptrend:
                # In uptrend: look for long breakouts above R1 with volume
                if curr_close > curr_r1 and volume_spike:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
            elif is_downtrend:
                # In downtrend: look for short breakdowns below S1 with volume
                if curr_close < curr_s1 and volume_spike:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 1.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 1.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals