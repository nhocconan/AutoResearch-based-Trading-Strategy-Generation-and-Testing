#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(2) extreme + volume confirmation
# KAMA adapts to market noise, reducing whipsaw in ranging markets
# RSI(2) < 10 for long, > 90 for short in trending markets (1d ADX > 25)
# Volume > 1.5x 20-period EMA confirms momentum
# Discrete sizing 0.25 minimizes fee churn. Target: 50-100 trades over 4 years.
# Works in bull/bear via regime filter and adaptive trend detection.

name = "1d_KAMA2_RSI_Volume_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period) with proper min_periods
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    
    plus_dm = high_1w.diff()
    minus_dm = low_1w.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_1w.sub(low_1w)
    tr2 = high_1w.sub(close_1w.shift(1)).abs()
    tr3 = low_1w.sub(close_1w.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align 1w ADX to 1d timeframe (completed 1w bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    
    # Calculate 1d KAMA (2-period efficiency, fast=2, slow=30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(2))
    volatility = close_s.diff().abs().rolling(window=2, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [np.nan] * len(close)
    if len(close) > 0:
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # Calculate 1d RSI(2) with proper min_periods
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            if adx_aligned[i] > 25:
                # Trending market: RSI(2) extreme in KAMA direction
                if close[i] > kama[i] and rsi[i] > 90 and volume_confirm:
                    # Uptrend: short on RSI > 90 (overbought)
                    signals[i] = -0.25
                    position = -1
                elif close[i] < kama[i] and rsi[i] < 10 and volume_confirm:
                    # Downtrend: long on RSI < 10 (oversold)
                    signals[i] = 0.25
                    position = 1
            else:
                # Ranging market: mean reversion at RSI extremes
                if rsi[i] < 10 and volume_confirm:
                    # Long at oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 90 and volume_confirm:
                    # Short at overbought
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: RSI > 50 OR ADX weakens (<20) OR volume drops
            if (rsi[i] > 50 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 OR ADX weakens (<20) OR volume drops
            if (rsi[i] < 50 or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals