#!/usr/bin/env python3
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
    
    # Get 4h HTF data once before loop (primary timeframe alignment)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 4h RSI(14) for momentum filter
    delta = pd.Series(df_4h['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = (100 - (100 / (1 + rs))).values
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Calculate 4h ATR(14) for volatility regime filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 12h HTF data for regime filter (choppiness index)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Choppiness Index(14) for regime detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3_12h = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_14_12h = np.where(
        (range_14 > 0) & (sum_tr_14 > 0),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50  # neutral value when range is zero
    )
    chop_14_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_14_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(chop_14_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: only trade when 4h ATR is elevated (> 0.3% of price)
        vol_regime = atr_14_4h_aligned[i] > 0.003 * close[i]
        
        # Trend filter: price relative to 4h EMA21
        bullish_trend = close[i] > ema_21_4h_aligned[i]
        bearish_trend = close[i] < ema_21_4h_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi_14_4h_aligned[i] < 70
        rsi_not_oversold = rsi_14_4h_aligned[i] > 30
        
        # Chop regime filter: 
        # CHOP > 61.8 = ranging market (favor mean reversion)
        # CHOP < 38.2 = trending market (favor trend following)
        # We'll use CHOP < 50 as trending bias for breakouts
        trending_regime = chop_14_12h_aligned[i] < 50
        
        # Long conditions:
        # 1. Bullish 4h trend (price > EMA21)
        # 2. RSI not overbought (< 70)
        # 3. Volatility regime active
        # 4. Trending regime from 12h chop
        if (bullish_trend and
            rsi_not_overbought and
            vol_regime and
            trending_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Bearish 4h trend (price < EMA21)
        # 2. RSI not oversold (> 30)
        # 3. Volatility regime active
        # 4. Trending regime from 12h chop
        elif (bearish_trend and
              rsi_not_oversold and
              vol_regime and
              trending_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA21_RSI14_VolRegime_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0