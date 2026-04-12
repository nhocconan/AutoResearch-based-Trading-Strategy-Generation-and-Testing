#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness filter
    # Captures institutional interest at key levels while avoiding choppy markets
    # Works in bull/bear by fading false breakouts in range and riding true breakouts in trend
    # Target: 20-40 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for choppiness calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Align 1d ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume_1d_aligned > (2.0 * vol_ma_20_1d_aligned)
    
    # Choppiness Index: CHOP > 61.8 = range (avoid), CHOP < 38.2 = trend (favor)
    # CHOP = 100 * log10(sum(TR over n) / (log10(n) * (max_high - min_low)))
    chop = np.full(len(df_1d), np.nan)
    lookback = 14
    for i in range(lookback, len(df_1d)):
        sum_tr = np.sum(tr[i-lookback+1:i+1])
        max_high = np.max(high_1d[i-lookback+1:i+1])
        min_low = np.min(low_1d[i-lookback+1:i+1])
        if max_high > min_low and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr) / (np.log10(lookback) * np.log10(max_high - min_low))
        else:
            chop[i] = 50.0  # neutral
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned < 61.8  # avoid extreme chop
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = np.full(n, np.nan)  # resistance
    camarilla_l4 = np.full(n, np.nan)  # support
    for i in range(1, n):
        # Use previous completed 1d bar
        idx_1d = i // 96  # approximate 1d bar index (96 * 15m = 24h, but we'll use alignment)
        # Better: get the actual 1d bar that closed before current 4h bar
        if i >= 96:  # need at least one full day of 4h bars
            # Find the 1d bar that corresponds to the day before current bar
            # We'll use the aligned HTF data approach but for levels we need the previous day's OHLC
            pass
    
    # Simpler approach: calculate Camarilla for each 1d bar then align
    camarilla_h4_1d = np.full(len(df_1d), np.nan)
    camarilla_l4_1d = np.full(len(df_1d), np.nan)
    camarilla_h3_1d = np.full(len(df_1d), np.nan)
    camarilla_l3_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            camarilla_h4_1d[i] = prev_close + 1.5 * range_
            camarilla_l4_1d[i] = prev_close - 1.5 * range_
            camarilla_h3_1d[i] = prev_close + 1.166 * range_
            camarilla_l3_1d[i] = prev_close - 1.166 * range_
        else:
            camarilla_h4_1d[i] = prev_close
            camarilla_l4_1d[i] = prev_close
            camarilla_h3_1d[i] = prev_close
            camarilla_l3_1d[i] = prev_close
    
    # Align Camarilla levels to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_spike[i]) if i < len(volume_spike) else True or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        breakout_long = close[i] > camarilla_h4_aligned[i] and volume_spike[i]
        breakout_short = close[i] < camarilla_l4_aligned[i] and volume_spike[i]
        
        # Mean reversion at inner levels in chop
        mean_revert_long = close[i] < camarilla_l3_aligned[i] and chop_aligned[i] > 50
        mean_revert_short = close[i] > camarilla_h3_aligned[i] and chop_aligned[i] > 50
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif mean_revert_long and position != 1 and chop_aligned[i] > 61.8:
            position = 1
            signals[i] = 0.20
        elif mean_revert_short and position != -1 and chop_aligned[i] > 61.8:
            position = -1
            signals[i] = -0.20
        elif position == 1 and (close[i] < camarilla_l3_aligned[i] or not chop_filter[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_h3_aligned[i] or not chop_filter[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0