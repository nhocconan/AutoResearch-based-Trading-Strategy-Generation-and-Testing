#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d EMA34 trend filter, volume spike (>2x average), and choppiness regime filter (CHOP > 50 for ranging markets). 
In ranging markets: price breaks above R3 with 1d uptrend and high volume → long; breaks below S3 with 1d downtrend and high volume → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe.
Requires BTC/ETH edge via 1d trend, volume, and regime filters; avoids SOL-only bias by requiring multi-factor alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.close.shift(1))), np.abs(low - np.close.shift(1)))).rolling(window).sum()
        max_high = pd.Series(high).rolling(window).max()
        min_low = pd.Series(low).rolling(window).min()
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(window)
        return chop.fillna(50).values  # fill NaN with 50 (neutral)
    
    # We need to calculate CHOP on 12h data, but we'll align it properly
    chop_raw = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1 for Camarilla calculation, 34 for EMA, 20 for volume, 14 for CHOP)
    start_idx = max(1, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels using previous day's OHLC
        # For 12h timeframe, previous day = previous 2 bars
        prev_1d_idx = i - 2
        if prev_1d_idx < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        prev_high = high[prev_1d_idx]
        prev_low = low[prev_1d_idx]
        prev_close = close[prev_1d_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Camarilla R3 and S3 levels
        R3 = prev_close + (range_val * 1.1 / 4)
        S3 = prev_close - (range_val * 1.1 / 4)
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        chop_val = chop_raw[i]
        
        # Skip if any data not ready
        if np.isnan(R3) or np.isnan(S3) or np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(chop_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Regime filter: Choppiness Index > 50 indicates ranging market (good for mean reversion/breakout)
        regime_filter = chop_val > 50
        
        # Long logic: price breaks above R3 with 1d uptrend, volume confirmation, and ranging regime
        long_condition = (close_val > R3) and (close_val > ema_val) and volume_confirmed and regime_filter
        # Short logic: price breaks below S3 with 1d downtrend, volume confirmation, and ranging regime
        short_condition = (close_val < S3) and (close_val < ema_val) and volume_confirmed and regime_filter
        
        # Exit logic: trend reversal or opposite breakout or regime change
        exit_long = (close_val < ema_val) or (close_val < S3) or (chop_val < 30)  # exit if strong trend
        exit_short = (close_val > ema_val) or (close_val > R3) or (chop_val < 30)  # exit if strong trend
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0