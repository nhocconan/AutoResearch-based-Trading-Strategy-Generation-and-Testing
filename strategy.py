#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h/1d regime filters to avoid whipsaws in trends.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for trend direction (EMA50 slope), 1d for volatility regime (ATR ratio).
- RSI(14): Long when RSI < 30 in bullish/low-vol regime, Short when RSI > 70 in bearish/low-vol regime.
- Regime filters: 
  * 4h EMA50 slope > 0 = bullish trend (favor longs), < 0 = bearish trend (favor shorts)
  * 1d ATR(10)/ATR(30) < 1.0 = low volatility (mean revert favorable), > 1.5 = high volatility (avoid)
- Volume confirmation: 1h volume > 1.5 * 20-period average to avoid low-liquidity false signals.
- Exit: Opposite RSI condition (RSI > 50 for long exit, RSI < 50 for short exit) or regime change.
- Signal size: 0.20 discrete to minimize fee drag.
- Works in bull markets by taking longs on pullbacks, in bear markets by taking shorts on bounces, and avoids ranging/choppy markets where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope of EMA50: (current - previous) / previous to get % change
    ema50_slope_4h = np.diff(ema50_4h, prepend=ema50_4h[0]) / np.where(ema50_4h == 0, 1e-10, ema50_4h)
    ema50_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_slope_4h)
    
    # Calculate 1d ATR(10) and ATR(30) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(10) and ATR(30)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio for volatility regime: <1.0 = low vol (good for mean reversion), >1.5 = high vol (avoid)
    atr_ratio = atr10 / atr30
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 20, 14)  # Need 50 for EMA50, 30 for ATR30, 20 for volume MA, 14 for RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_slope_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_rsi = rsi[i]
        curr_volume = volume[i]
        
        # Regime filters
        bullish_trend = ema50_slope_4h_aligned[i] > 0  # 4h EMA50 rising = bullish
        bearish_trend = ema50_slope_4h_aligned[i] < 0  # 4h EMA50 falling = bearish
        low_volatility = atr_ratio_aligned[i] < 1.0    # 1d ATR ratio < 1.0 = low vol (mean revert favorable)
        high_volatility = atr_ratio_aligned[i] > 1.5   # 1d ATR ratio > 1.5 = high vol (avoid trading)
        
        # Volume confirmation: current volume > 1.5 * 20-period average (both 1h and 1d)
        volume_confirm_1h = curr_volume > 1.5 * vol_ma_20[i]
        volume_confirm_1d = not np.isnan(vol_ma_20_1d_aligned[i]) and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        volume_confirm = volume_confirm_1h and volume_confirm_1d
        
        # Exit conditions
        if position != 0:
            # Exit long: RSI > 50 OR regime becomes unfavorable (high vol or trend flip)
            if position == 1:
                if curr_rsi > 50 or high_volatility or (bullish_trend and not bearish_trend and curr_rsi > 60):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: RSI < 50 OR regime becomes unfavorable
            elif position == -1:
                if curr_rsi < 50 or high_volatility or (bearish_trend and not bullish_trend and curr_rsi < 40):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: RSI mean reversion with regime and volume filters
        if position == 0:
            # Long: RSI < 30 (oversold) AND bullish trend OR low volatility AND volume confirmation
            long_condition = (curr_rsi < 30 and 
                            ((bullish_trend and low_volatility) or low_volatility) and  # Bullish trend OR low vol
                            volume_confirm)
            
            # Short: RSI > 70 (overbought) AND bearish trend OR low volatility AND volume confirmation
            short_condition = (curr_rsi > 70 and 
                             ((bearish_trend and low_volatility) or low_volatility) and  # Bearish trend OR low vol
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_RSI14_MeanReversion_4hEMA50Trend_1dATRRegime_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0