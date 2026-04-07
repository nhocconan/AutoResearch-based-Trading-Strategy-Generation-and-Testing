#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily ADX trend strength with weekly RSI extremes for mean reversion
# Uses ADX(14) to filter trending vs ranging markets, weekly RSI(14) for overbought/oversold signals,
# and price position relative to EMA(50) for entry timing. Designed for low trade frequency
# (target: 15-25 trades/year) to minimize fee drag. Works in trending markets via ADX filter
# and in ranging markets via RSI mean reversion at extremes.

name = "daily_adx_weekly_rsi_extremes_v1"
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
    
    # Weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ADX(14) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    atr = np.zeros(n)
    atr[tr_period-1] = np.mean(tr[:tr_period])
    for i in range(tr_period, n):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    plus_dm_smooth[tr_period-1] = np.sum(plus_dm[:tr_period])
    minus_dm_smooth[tr_period-1] = np.sum(minus_dm[:tr_period])
    
    for i in range(tr_period, n):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / tr_period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / tr_period) + minus_dm[i]
    
    # Calculate +DI and -DI
    for i in range(tr_period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Smooth DX to get ADX
    adx = np.zeros(n)
    adx[2*tr_period-2] = np.mean(dx[tr_period-1:2*tr_period-1])
    for i in range(2*tr_period-1, n):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
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
    
    # EMA(50) for entry timing
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema50[i])):
            signals[i] = 0.0
            continue
        
        # ADX threshold for trending market
        trending = adx[i] > 25
        
        # Weekly RSI extremes for mean reversion
        oversold = rsi_1w_aligned[i] < 30
        overbought = rsi_1w_aligned[i] > 70
        
        # Price relative to EMA(50)
        price_above_ema = close[i] > ema50[i]
        price_below_ema = close[i] < ema50[i]
        
        # Long conditions:
        # 1. In trending market AND price above EMA(50) (trend following)
        # 2. OR weekly RSI oversold (mean reversion)
        if (trending and price_above_ema) or oversold:
            signals[i] = 0.25
        # Short conditions:
        # 1. In trending market AND price below EMA(50) (trend following)
        # 2. OR weekly RSI overbought (mean reversion)
        elif (trending and price_below_ema) or overbought:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals