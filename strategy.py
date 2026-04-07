#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily ADX trend filter with weekly RSI extremes and volume confirmation
# Uses ADX(14) to identify trending markets, weekly RSI(14) for overbought/oversold extremes,
# and volume spike confirmation to enter trades. Designed for low trade frequency
# (target: 15-25 trades/year) to minimize fee drift. Works in trending markets via
# ADX filter and in ranging markets via mean reversion at RSI extremes with volume confirmation.

name = "daily_adx_weekly_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ADX(14) for trend strength
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) != 0, dx, 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume spike detection (volume > 1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Extreme RSI conditions for mean reversion
        oversold = rsi_1w_aligned[i] < 30
        overbought = rsi_1w_aligned[i] > 70
        
        # Volume confirmation
        vol_conf = vol_spike[i] if not np.isnan(vol_spike[i]) else False
        
        # Long conditions: trend up + oversold OR ranging + oversold with volume
        if (trending and plus_di[i] > minus_di[i] and oversold) or \
           (~trending and oversold and vol_conf):
            signals[i] = 0.25
        # Short conditions: trend down + overbought OR ranging + overbought with volume
        elif (trending and minus_di[i] > plus_di[i] and overbought) or \
             (~trending and overbought and vol_conf):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals