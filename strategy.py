#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Volume Confirmation with 1d Trend Filter
# ADX(14) > 25 indicates strong trend (avoids choppy markets)
# Volume > 1.5x 20-period average confirms institutional participation
# 1d EMA(50) provides higher timeframe trend bias to avoid counter-trend trades
# Enter long when: +DI > -DI (bullish momentum) + ADX > 25 + price > 1d EMA50 + volume confirmation
# Enter short when: -DI > +DI (bearish momentum) + ADX > 25 + price < 1d EMA50 + volume confirmation
# Exit when ADX < 20 (trend weakening) or opposite DI crossover
# Works in bull/bear as ADX filters range markets and 1d EMA adapts to trend
# Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ADX calculation (14 periods)
    adx_len = 14
    if len(high) < adx_len + 1:
        return np.zeros(n)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(adx_len * 2, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or 
            np.isnan(plus_di[i]) or
            np.isnan(minus_di[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weakening_trend = adx[i] < 20
        
        # Directional bias
        bullish_momentum = plus_di[i] > minus_di[i]
        bearish_momentum = minus_di[i] > plus_di[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: bullish momentum + strong trend + above 1d EMA + volume
            if (bullish_momentum and 
                strong_trend and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: bearish momentum + strong trend + below 1d EMA + volume
            elif (bearish_momentum and 
                  strong_trend and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: weakening trend or bearish momentum shift
            if weakening_trend or (minus_di[i] > plus_di[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: weakening trend or bullish momentum shift
            if weakening_trend or (plus_di[i] > minus_di[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ADX_Volume_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0