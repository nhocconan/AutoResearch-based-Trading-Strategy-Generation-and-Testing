#!/usr/bin/env python3
# Hypothesis: 1h RSI(14) mean reversion with 4h ADX(14) regime filter and volume confirmation.
# Long when RSI < 30 (oversold) AND 4h ADX < 25 (range/chop regime) AND volume > 1.3x 20-bar average.
# Short when RSI > 70 (overbought) AND 4h ADX < 25 (range/chop regime) AND volume > 1.3x average.
# Exit when RSI crosses back above 50 (for longs) or below 50 (for shorts).
# Uses discrete position sizing 0.20. Target: 60-150 total trades over 4 years on 1h timeframe.
# ADX regime filter ensures we only mean revert in choppy/range markets, avoiding trending whipsaws.
# Volume confirmation validates mean reversion strength. RSI 50 exit provides clear, objective stop.
# RSI mean reversion works well in ranging markets (2025 BTC/ETH bear/range) and avoids strong trends.

name = "1h_RSI14_4hADX14_Range_MeanReversion_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    lookback = 20  # for volume average and RSI calculations
    rsi_period = 14
    adx_period = 14
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral RSI when no data
    
    # Get 4h data for ADX regime filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < adx_period * 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    # True Range
    tr1 = pd.Series(high_4h).diff().abs()
    tr2 = (pd.Series(high_4h) - pd.Series(close_4h).shift()).abs()
    tr3 = (pd.Series(low_4h) - pd.Series(close_4h).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean()
    
    # Directional Movement
    up_move = pd.Series(high_4h).diff()
    down_move = -pd.Series(low_4h).diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed DM
    plus_di = 100 * (plus_dm.ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean() / atr)
    
    # DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    adx = dx.ewm(alpha=1/adx_period, adjust=False, min_periods=adx_period).mean()
    adx_values = adx.values
    
    # Align 4h ADX to 1h timeframe (wait for 4h bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_values)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold (<30) AND 4h ADX < 25 (range/chop) AND volume spike
            if (rsi[i] < 30 and 
                adx_aligned[i] < 25 and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI overbought (>70) AND 4h ADX < 25 (range/chop) AND volume spike
            elif (rsi[i] > 70 and 
                  adx_aligned[i] < 25 and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses back above 50 (mean reversion complete)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI crosses back below 50 (mean reversion complete)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals