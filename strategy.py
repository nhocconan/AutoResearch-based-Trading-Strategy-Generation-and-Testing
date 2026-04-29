#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter with volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to balance return and fee drag
# Target: 12-37 trades/year on 6h timeframe to avoid fee drag while capturing strong momentum shifts
# Using 12h EMA50 for trend filter and 12h ADX > 25 for regime filter (only trade in trending markets)
# Volume confirmation ensures breakouts have strong participation
# Works in bull markets via strong Bull Power with 12h uptrend
# Works in bear markets via strong Bear Power with 12h downtrend

name = "6h_ElderRay_12hADX_Regime_VolumeConfirm_v1"
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
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h ADX for regime filter (only trade when ADX > 25)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_first = np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])
    tr_12h = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus_first = np.maximum(high_12h[0] - high_12h[0], 0)  # Always 0 for first bar
    dm_minus_first = np.maximum(low_12h[0] - low_12h[0], 0)   # Always 0 for first bar
    dm_plus_12h = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus_12h = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / atr_12h
    di_minus = 100 * dm_minus_smoothed / atr_12h
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate ATR for stoploss (using 14-period on LTF)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Elder Ray components: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = low - ema_13   # Low - EMA(13)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_adx_12h = adx_12h_aligned[i]
        curr_atr = atr[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        # Skip if ADX is not available
        if np.isnan(curr_adx_12h):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Regime filter: only trade when ADX > 25 (trending market)
        in_trending_regime = curr_adx_12h > 25
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
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
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bull Power rising (current > previous) AND price > 12h EMA50 AND volume spike AND trending regime
            if (curr_bull_power > 0 and 
                i > start_idx and curr_bull_power > bull_power[i-1] and 
                curr_close > curr_ema_12h and vol_spike and in_trending_regime):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Bear Power < 0 AND Bear Power falling (current < previous) AND price < 12h EMA50 AND volume spike AND trending regime
            elif (curr_bear_power < 0 and 
                  i > start_idx and curr_bear_power < bear_power[i-1] and 
                  curr_close < curr_ema_12h and vol_spike and in_trending_regime):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals