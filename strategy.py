#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h1d_TrendMomentum - Long when 1h momentum aligns with 4h/1d uptrend and volume surge; short when momentum aligns with downtrend and volume surge.
# Uses 4h EMA20 for trend, 1d ADX for trend strength, and 1h ROC + RSI for momentum entry. Volume > 1.5x 20-period average confirms.
# Designed for 15-25 trades/year per symbol by requiring multi-timeframe alignment and volume confirmation.
# Works in bull/bear via trend filters and momentum exhaustion exits.
name = "1h_4h1d_TrendMomentum"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for ADX trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 1h momentum indicators
    roc_10 = ((pd.Series(close).pct_change(10)) * 100).values
    rsi_14 = compute_rsi(close, 14)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(roc_10[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_20_4h = ema_20_4h_aligned[i]
        adx = adx_1d_aligned[i]
        roc = roc_10[i]
        rsi = rsi_14[i]
        
        # Volume surge condition
        volume_surge = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: 4h uptrend, strong trend (ADX > 25), bullish momentum (ROC > 0, RSI > 50)
            if (price > ema_20_4h and adx > 25 and roc > 0 and rsi > 50 and volume_surge):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, strong trend (ADX > 25), bearish momentum (ROC < 0, RSI < 50)
            elif (price < ema_20_4h and adx > 25 and roc < 0 and rsi < 50 and volume_surge):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend weakening (ADX < 20) or momentum exhaustion (ROC < 0 and RSI < 50)
            if adx < 20 or (roc < 0 and rsi < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend weakening (ADX < 20) or momentum exhaustion (ROC > 0 and RSI > 50)
            if adx < 20 or (roc > 0 and rsi > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

def compute_rsi(prices, period=14):
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values