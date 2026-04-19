#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Engulfing_Candle_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d ATR for volatility filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w trend direction (using EMA50)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d ATR and 1w EMA to 6h timeframe
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_6h[i]) or np.isnan(ema_50_1w_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Current candle body
        body = abs(close[i] - open_[i])
        upper_wick = high[i] - max(close[i], open_[i])
        lower_wick = min(close[i], open_[i]) - low[i]
        
        # Engulfing candle conditions
        bullish_engulfing = (
            close[i] > open_[i] and  # bullish candle
            open_[i] <= close[i-1] and  # opens below or at previous close
            close[i] >= open_[i-1] and  # closes above or at previous open
            body > (upper_wick + lower_wick) * 0.5  # significant body
        )
        
        bearish_engulfing = (
            close[i] < open_[i] and  # bearish candle
            open_[i] >= close[i-1] and  # opens above or at previous close
            close[i] <= open_[i-1] and  # closes below or at previous open
            body > (upper_wick + lower_wick) * 0.5  # significant body
        )
        
        # Volume spike: current volume > 1.8x average
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # ATR filter: only trade when volatility is elevated
        vol_filter = atr_1d_6h[i] > np.nanmedian(atr_1d_6h[max(0, i-50):i+1])
        
        # Trend filter: use 1w EMA50
        uptrend = close[i] > ema_50_1w_6h[i]
        downtrend = close[i] < ema_50_1w_6h[i]
        
        if position == 0:
            # Long: Bullish engulfing + volume spike + volatility filter + uptrend
            if bullish_engulfing and volume_spike and vol_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing + volume spike + volatility filter + downtrend
            elif bearish_engulfing and volume_spike and vol_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish engulfing or price drops below 1w EMA50
            if bearish_engulfing or close[i] < ema_50_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish engulfing or price rises above 1w EMA50
            if bullish_engulfing or close[i] > ema_50_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals