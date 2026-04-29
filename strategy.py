#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Uses tight entry conditions (breakout of Camarilla R3/S3 levels) combined with
# 1d EMA34 trend filter and volume confirmation to reduce false breakouts.
# Targets 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following 1d trend while capturing 4h momentum.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels using previous day's OHLC
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # We use previous day's data to avoid look-ahead
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (20*4h = ~3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for EMA34 (34+1 for shift) and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_open = open_[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: close breaks above R3 AND bullish regime
                if curr_close > curr_r3 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: close breaks below S3 AND bearish regime
                elif curr_close < curr_s3 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: close breaks below R3 OR regime changes to bearish
            if curr_close < curr_r3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: close breaks above S3 OR regime changes to bullish
            if curr_close > curr_s3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals