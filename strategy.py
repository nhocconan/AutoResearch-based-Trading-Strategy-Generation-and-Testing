#!/usr/bin/env python3
# 1d_Chaikin_Money_Flow_RSI_Trend_Filter
# Hypothesis: Daily Chaikin Money Flow (CMF) with RSI and trend filter to capture strong momentum moves.
# Uses 1d CMF for money flow direction, 14-period RSI for overbought/oversold conditions, and 50-day EMA for trend.
# Designed for low trade frequency (10-25/year) on 1d timeframe to minimize fee decay while capturing sustained trends in BTC/ETH/SOL.

name = "1d_Chaikin_Money_Flow_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly trend: EMA50 ---
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily Chaikin Money Flow (CMF) over 20 periods ---
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # --- Daily RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(cmf[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from weekly EMA50
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: CMF > 0.1 (strong buying pressure), RSI < 70 (not overbought), bullish weekly trend
            if cmf[i] > 0.1 and rsi[i] < 70 and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.1 (strong selling pressure), RSI > 30 (not oversold), bearish weekly trend
            elif cmf[i] < -0.1 and rsi[i] > 30 and bearish_trend:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: CMF turns negative or RSI > 70 (overbought)
                if cmf[i] < 0 or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: CMF turns positive or RSI < 30 (oversold)
                if cmf[i] > 0 or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals