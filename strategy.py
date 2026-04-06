#!/usr/bin/env python3
"""
1h Volume-Weighted RSI with 4h Trend and 1d Volatility Regime
Hypothesis: In ranging markets (low volatility), buy oversold RSI with volume confirmation;
in trending markets (high volatility), sell overbought RSI. The 4h EMA determines trend direction
while 1d ATR percentile determines volatility regime. This adapts to both bull and bear markets
by switching between mean reversion and momentum based on volatility. Volume confirmation
reduces false signals. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vw_rsi_4h_trend_1d_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = np.full(n, np.nan)
        roll_down = np.full(n, np.nan)
        if len(up) >= 14:
            roll_up[13] = np.mean(up[:14])
            roll_down[13] = np.mean(down[:14])
            for i in range(14, n):
                roll_up[i] = (roll_up[i-1] * 13 + up[i-1]) / 14
                roll_down[i] = (roll_down[i-1] * 13 + down[i-1]) / 14
        rs = np.where(roll_down != 0, roll_up / roll_down, 0)
        rsi = 100 - (100 / (1 + rs))
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 34-period EMA on 4h for trend
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 34:
        ema_4h[33] = np.mean(close_4h[:34])
        for i in range(34, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 32) / 34
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Get 1d data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        tr_1d = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        if len(tr_1d) > 0:
            atr_1d[13] = np.mean(tr_1d[:14])
            for i in range(14, len(atr_1d)):
                atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i-1]) / 14
    
    # ATR percentile (50-period lookback) for volatility regime
    atr_percentile = np.full(len(df_1d), np.nan)
    if len(atr_1d) >= 50:
        for i in range(49, len(atr_1d)):
            window = atr_1d[i-49:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                rank = (np.sum(valid <= atr_1d[i]) / len(valid)) * 100
                atr_percentile[i] = rank
    
    # Regime: 1 if high volatility (percentile > 50), -1 if low volatility (percentile <= 50)
    vol_regime = np.where(atr_percentile > 50, 1, -1)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 34, 20)
    
    for i in range(start, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(trend_4h_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(vol_regime_aligned[i]) or
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit conditions depend on volatility regime
            if vol_regime_aligned[i] == 1:  # high volatility - momentum
                # Exit: RSI > 60 or trend turns down
                if (rsi[i] > 60 or trend_4h_aligned[i] == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # low volatility - mean reversion
                # Exit: RSI > 50 or price drops 2*ATR below entry
                if (rsi[i] > 50 or close[i] < entry_price - 2.0 * atr[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:  # short position
            # Exit conditions depend on volatility regime
            if vol_regime_aligned[i] == 1:  # high volatility - momentum
                # Exit: RSI < 40 or trend turns up
                if (rsi[i] < 40 or trend_4h_aligned[i] == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:  # low volatility - mean reversion
                # Exit: RSI < 50 or price rises 2*ATR above entry
                if (rsi[i] < 50 or close[i] > entry_price + 2.0 * atr[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Look for entries based on volatility regime
            if vol_regime_aligned[i] == 1:  # high volatility - momentum
                # Long: RSI < 40 in uptrend with volume
                if (rsi[i] < 40 and trend_4h_aligned[i] == 1 and volume_filter):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 60 in downtrend with volume
                elif (rsi[i] > 60 and trend_4h_aligned[i] == -1 and volume_filter):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:  # low volatility - mean reversion
                # Long: RSI < 30 with volume (oversold)
                if (rsi[i] < 30 and volume_filter):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 with volume (overbought)
                elif (rsi[i] > 70 and volume_filter):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
    
    return signals