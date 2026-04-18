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
    
    # Get daily data for calculations (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14-period)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ADX (14-period)
    up_move = df_1d['high'] - np.roll(df_1d['high'], 1)
    down_move = np.roll(df_1d['low'], 1) - df_1d['low']
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_adx = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_adx = pd.Series(tr_adx).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_adx + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_adx + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h SMA (50-period) for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(sma_50[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 20-period average
        if i >= 20:
            atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
            atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
            vol_filter = not np.isnan(atr_ma_1d_aligned[i]) and atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        else:
            vol_filter = False
        
        # Trend filter: ADX > 25 for trending market
        trend_filter = adx_aligned[i] > 25
        
        trade_allowed = vol_filter and trend_filter
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > SMA50 (uptrend)
            if trade_allowed and rsi[i] < 30 and close[i] > sma_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price < SMA50 (downtrend)
            elif trade_allowed and rsi[i] > 70 and close[i] < sma_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 or price < SMA50
            if rsi[i] > 50 or close[i] < sma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 or price > SMA50
            if rsi[i] < 50 or close[i] > sma_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_SMA50_ATR_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0