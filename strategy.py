#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume spike.
# Enters long when price breaks above R3 level with 1d bullish trend (ADX > 25 and +DI > -DI) and volume > 1.8x MA20.
# Enters short when price breaks below S3 level with 1d bearish trend (ADX > 25 and -DI > +DI) and volume > 1.8x MA20.
# Exits when price crosses the 12h EMA20 (adaptive mean reversion).
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~12-37/year) to work in both bull and bear markets by requiring strong volume confirmation, trend alignment, and avoiding choppy markets.

name = "12h_Camarilla_R3S3_Breakout_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (based on previous 1d bar)
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for ADX trend filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    # Prepend first values to maintain length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align ADX, +DI, -DI to LTF
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Get 12h data for exit condition (EMA20)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume filter: current volume > 1.8x 20-period average (balanced to reduce trades)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or \
           np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with 1d bullish trend (ADX>25 and +DI>-DI) and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with 1d bearish trend (ADX>25 and -DI>+DI) and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 12h EMA20 (mean reversion in range)
            if close[i] < ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 12h EMA20 (mean reversion in range)
            if close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals