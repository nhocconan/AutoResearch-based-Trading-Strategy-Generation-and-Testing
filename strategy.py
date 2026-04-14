#!/usr/bin/env python3
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
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA 21 for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).values
    
    # Daily ATR (14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily RSI (14) for mean reversion signal
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Get aligned weekly EMA
        ema_21_1w_i = align_htf_to_ltf(prices, df_1w, ema_21_1w)[i]
        # Get aligned daily ATR and RSI
        atr_14_1d_i = align_htf_to_ltf(prices, df_1d, atr_14_1d)[i]
        rsi_14_1d_i = align_htf_to_ltf(prices, df_1d, rsi_14_1d)[i]
        
        if np.isnan(ema_21_1w_i) or np.isnan(atr_14_1d_i) or np.isnan(rsi_14_1d_i):
            continue
        
        # Volatility filter: ATR normalized by price
        atr_norm = atr_14_1d_i / close[i] if close[i] > 0 else np.nan
        low_vol = atr_norm < 0.02  # 2% ATR threshold
        
        # Mean reversion: RSI oversold/overbought
        rsi_oversold = rsi_14_1d_i < 30
        rsi_overbought = rsi_14_1d_i > 70
        
        # Trend filter: price relative to weekly EMA21
        price_above_weekly_ema = close[i] > ema_21_1w_i
        price_below_weekly_ema = close[i] < ema_21_1w_i
        
        # Long: oversold RSI + low volatility + price above weekly EMA (bullish mean reversion)
        if position == 0 and low_vol and rsi_oversold and price_above_weekly_ema:
            position = 1
            signals[i] = position_size
        # Short: overbought RSI + low volatility + price below weekly EMA (bearish mean reversion)
        elif position == 0 and low_vol and rsi_overbought and price_below_weekly_ema:
            position = -1
            signals[i] = -position_size
        
        # Exit: RSI returns to neutral zone (40-60)
        elif position != 0:
            rsi_exit = 40 <= rsi_14_1d_i <= 60
            if rsi_exit:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA21_RSI14_MeanReversion_LowVol_Filter"
timeframe = "1d"
leverage = 1.0