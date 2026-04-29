#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w ADX regime filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND 1w ADX > 25 (trending) AND volume > 1.5x 20-period average
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 12-25 trades/year on 6h timeframe (~50-100 total over 4 years)
# Works in bull markets via long entries with 1w uptrend
# Works in bear markets via short entries with 1w downtrend
# Uses 1w timeframe for regime filter to avoid whipsaws in ranging markets

name = "6h_ElderRay_1wADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA13 for Elder Ray
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1w ADX for regime filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w_arr[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w_arr[:-1])
    tr_first = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w_arr[0]), np.abs(low_1w[0] - close_1w_arr[0])])
    tr_1w = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus_first = np.maximum(high_1w[0] - high_1w[0], 0)  # Always 0 for first bar
    dm_minus_first = np.maximum(low_1w[0] - low_1w[0], 0)   # Always 0 for first bar
    dm_plus = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed values
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate ATR for stoploss (using 14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_13 = ema_13_1w_aligned[i]
        curr_adx = adx_1w_aligned[i]
        curr_atr = atr[i]
        
        # Elder Ray components
        bull_power = curr_high - curr_ema_13
        bear_power = curr_low - curr_ema_13
        
        # Skip if indicators are not available
        if np.isnan(curr_ema_13) or np.isnan(curr_adx) or np.isnan(curr_atr):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND ADX > 25 (trending) AND volume spike
            if bull_power > 0 and bear_power < 0 and curr_adx > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND ADX > 25 (trending) AND volume spike
            elif bear_power < 0 and bull_power > 0 and curr_adx > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals