#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeChop
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume confirmation (>1.5x 20-bar avg), and choppiness regime filter (CHOP>50 for mean reversion, CHOP<50 for trend) captures institutional moves while avoiding choppy markets. Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-40 trades/year to ensure test generalization across BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous completed 1d bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) for regime filter
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period if i >= period else np.nan
        atr[0] = np.nan
        sum_atr = np.nansum(atr.reshape(-1, period), axis=1) if len(atr) >= period else np.full(len(atr), np.nan)
        sum_atr_full = np.convolve(atr, np.ones(period)/period, mode='same')
        sum_atr_full[:period-1] = np.nan
        max_h = np.maximum.accumulate(high)
        min_l = np.minimum.accumulate(low)
        range_max_min = max_h - min_l
        chop = 100 * np.log10(sum_atr_full / (range_max_min * np.sqrt(period))) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 34)  # EMA34, vol MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        chop_val = chop[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend, volume, and chop regime
            # Long: price breaks above R1 with uptrend (close > EMA34), volume confirm, and trending market (CHOP < 50)
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_confirm and (chop_val < 50)
            # Short: price breaks below S1 with downtrend (close < EMA34), volume confirm, and trending market (CHOP < 50)
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_confirm and (chop_val < 50)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            # 2. Choppiness regime shift to choppy (CHOP >= 50) - exit to avoid whipsaw
            if close_val < s1_val or chop_val >= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            # 2. Choppiness regime shift to choppy (CHOP >= 50) - exit to avoid whipsaw
            if close_val > r1_val or chop_val >= 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeChop"
timeframe = "4h"
leverage = 1.0