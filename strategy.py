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
    
    # Get 4h HTF data once before loop (primary trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 12h HTF data for regime filter (choppiness)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Choppiness Index (CHOP)
    hl_range = df_12h['high'].values - df_12h['low'].values
    atr_12h = pd.Series(np.maximum(
        hl_range,
        np.maximum(
            np.abs(df_12h['high'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]])),
            np.abs(df_12h['low'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]]))
        )
    )).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    sum_atr_12h = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_h_12h = pd.Series(df_12h['high'].values).rolling(window=14, min_periods=14).max().values
    min_l_12h = pd.Series(df_12h['low'].values).rolling(window=14, min_periods=14).min().values
    chop_12h = 100 * np.log10(sum_atr_12h / (max_h_12h - min_l_12h + 1e-10)) / np.log10(14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 12h ATR for volatility filter
    tr_12h = pd.Series(np.maximum(
        hl_range,
        np.maximum(
            np.abs(df_12h['high'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]])),
            np.abs(df_12h['low'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]]))
        )
    )).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, tr_12h)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20_12h + 1e-10)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: CHOP > 50 indicates choppy market (mean reversion favorable)
        in_chop = chop_12h_aligned[i] > 50
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_12h_aligned[i] > 0.005 * close[i]
        
        # Volume confirmation
        vol_confirm = vol_ratio_12h_aligned[i] > 1.3
        
        # Mean reversion conditions in choppy market
        if in_chop and vol_filter and vol_confirm:
            # Calculate 12h RSI for mean reversion signals
            delta = pd.Series(df_12h['close'].values).diff().values
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
            avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            rsi_12h = 100 - (100 / (1 + rs))
            rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
            
            if not np.isnan(rsi_12h_aligned[i]):
                # Oversold: RSI < 30 -> long
                if rsi_12h_aligned[i] < 30:
                    signals[i] = 0.25
                # Overbought: RSI > 70 -> short
                elif rsi_12h_aligned[i] > 70:
                    signals[i] = -0.25
        
        # Trending regime: CHOP <= 50 indicates trending market
        else:
            # Trend following: price above/below 4h EMA34 with volume
            if close[i] > ema_34_4h_aligned[i] and vol_confirm:
                signals[i] = 0.25
            elif close[i] < ema_34_4h_aligned[i] and vol_confirm:
                signals[i] = -0.25
    
    return signals

name = "12h_CHOP_Regime_4h_EMA34_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0