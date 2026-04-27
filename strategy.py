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
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 4h Bollinger Bands (20, 2) for volatility and mean reversion
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower.values)
    
    # Calculate 4h volume moving average for confirmation
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h.values)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_not_extreme = (rsi_14_1d_aligned[i] > 30) and (rsi_14_1d_aligned[i] < 70)
        
        # Volume filter: current 4h volume above average
        volume_filter = vol_ma_4h_aligned[i] > 0 and volume[i] > vol_ma_4h_aligned[i] * 1.2
        
        # Mean reversion signals: price touches Bollinger Bands
        touch_upper = close[i] >= bb_upper_aligned[i]
        touch_lower = close[i] <= bb_lower_aligned[i]
        
        # Long conditions: bullish trend + RSI not extreme + volume + touch lower BB (mean reversion long)
        long_condition = (price_above_ema and 
                         rsi_not_extreme and 
                         volume_filter and 
                         touch_lower)
        
        # Short conditions: bearish trend + RSI not extreme + volume + touch upper BB (mean reversion short)
        short_condition = (price_below_ema and 
                          rsi_not_extreme and 
                          volume_filter and 
                          touch_upper)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: touch opposite band or trend reversal
        elif position == 1 and (touch_upper or not price_above_ema):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (touch_lower or not price_below_ema):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA50_RSI14_4hBB_MeanReversion"
timeframe = "4h"
leverage = 1.0