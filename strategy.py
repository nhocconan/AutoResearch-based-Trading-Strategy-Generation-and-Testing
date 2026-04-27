#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(20) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate 4-period RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/4, min_periods=4).mean()
    avg_loss = loss.ewm(alpha=1/4, min_periods=4).mean()
    rs = avg_gain / avg_loss
    rsi_4 = 100 - (100 / (1 + rs))
    rsi_4 = rsi_4.fillna(50).values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_20_1d_aligned[i]) or
            np.isnan(rsi_4[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_20_1d_aligned[i] > 0 and atr_20_1d_aligned[i] < np.median(atr_20_1d_aligned[:i+1]) * 2.5
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi_4[i] > 30) and (rsi_4[i] < 70)
        
        # Volume filter: above average volume (using 20-period MA)
        vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
        if np.isnan(vol_ma_20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > vol_ma_20_1d_aligned[i] * 1.3
        
        # Long conditions: bullish trend + volatility filter + RSI filter + volume spike
        long_condition = (price_above_ema and vol_filter and rsi_filter and vol_spike)
        
        # Short conditions: bearish trend + volatility filter + RSI filter + volume spike
        short_condition = (price_below_ema and vol_filter and rsi_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_4[i] > 75):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_4[i] < 25):
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

name = "4h_DailyEMA50_RSI4_VolumeFilter_Session"
timeframe = "4h"
leverage = 1.0