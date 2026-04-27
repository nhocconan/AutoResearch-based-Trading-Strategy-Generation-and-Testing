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
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14) for momentum/overbought-oversold
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_values = atr.values
    
    # Calculate daily volume moving average for confirmation
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_values[i]) or
            np.isnan(atr_values[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Momentum filter: RSI not in extreme overbought/oversold
        rsi_not_overbought = rsi_values[i] < 70
        rsi_not_oversold = rsi_values[i] > 30
        
        # Volatility filter: avoid excessively volatile periods
        vol_filter = atr_values[i] < (np.mean(atr_values[max(0, i-50):i+1]) * 1.5)
        
        # Volume filter: current volume above average
        volume_filter = vol_ma_10[i] > 0 and volume[i] > vol_ma_10[i] * 0.5
        
        # Long conditions: bullish trend + moderate momentum + volume + volatility filter
        long_condition = (price_above_ema and 
                         rsi_not_overbought and
                         volume_filter and
                         vol_filter)
        
        # Short conditions: bearish trend + moderate momentum + volume + volatility filter
        short_condition = (price_below_ema and 
                          rsi_not_oversold and
                          volume_filter and
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_values[i] >= 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_values[i] <= 30):
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

name = "1d_EMA50_1w_RSI14_VolumeFilter"
timeframe = "1d"
leverage = 1.0