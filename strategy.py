#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX + RSI mean reversion with 1d trend filter and volume confirmation
# Uses ADX(14) > 25 to identify trending markets, then enters mean reversion trades when RSI(14) is extreme
# In uptrend (1d EMA50): buy when RSI < 30, sell when RSI > 70
# In downtrend (1d EMA50): sell short when RSI > 70, buy to cover when RSI < 30
# Volume confirmation ensures institutional participation
# Designed for low trade frequency in both bull and bear markets
# Target: 60-120 total trades over 4 years = 15-30/year

name = "4h_ADX_RSI_MeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[np.nan], rsi.values])

def adx(high, low, close, period=14):
    """Average Directional Index"""
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (pd.Series(tr).ewm(alpha=1/period, adjust=False).mean() + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (pd.Series(tr).ewm(alpha=1/period, adjust=False).mean() + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
    return np.concatenate([[np.nan], adx.values])

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate indicators on 4h data
    rsi_val = rsi(close, 14)
    adx_val = adx(high, low, close, 14)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(adx_val[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        rsi_val_i = rsi_val[i]
        adx_val_i = adx_val[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: ADX > 25 (trending), RSI < 30 (oversold), uptrend, volume spike
            if (adx_val_i > 25 and rsi_val_i < 30 and 
                close[i] > ema50_1d_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: ADX > 25 (trending), RSI > 70 (overbought), downtrend, volume spike
            elif (adx_val_i > 25 and rsi_val_i > 70 and 
                  close[i] < ema50_1d_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) OR trend breaks
            if rsi_val_i > 50 or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) OR trend breaks
            if rsi_val_i < 50 or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals