#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_chaikin_money_flow_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d CMF (Chaikin Money Flow) with period 20
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # CMF(20) = sum(MFV, 20) / sum(volume, 20)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf_20_1d = np.where(vol_sum == 0, 0, mfv_sum / vol_sum)
    
    # Calculate 1d RSI(14) for overbought/oversold filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d ADX(14) for trend strength filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    plus_di_14_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values / atr_14_1d
    minus_di_14_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values / atr_14_1d
    dx = np.where((plus_di_14_1d + minus_di_14_1d) == 0, 0, 
                  100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d))
    adx_14_1d = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align all 1d indicators to 12h timeframe
    cmf_20_1d_aligned = align_htf_to_ltf(prices, df_1d, cmf_20_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data for calculations
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cmf_20_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        cmf = cmf_20_1d_aligned[i]
        rsi = rsi_14_1d_aligned[i]
        adx = adx_14_1d_aligned[i]
        
        # Long when CMF > 0.1 (bullish money flow), RSI < 70 (not overbought), and ADX > 20 (trending)
        long_condition = (cmf > 0.1) and (rsi < 70) and (adx > 20)
        
        # Short when CMF < -0.1 (bearish money flow), RSI > 30 (not oversold), and ADX > 20 (trending)
        short_condition = (cmf < -0.1) and (rsi > 30) and (adx > 20)
        
        # Exit when CMF crosses zero (money flow reversal)
        exit_long = cmf < 0
        exit_short = cmf > 0
        
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals