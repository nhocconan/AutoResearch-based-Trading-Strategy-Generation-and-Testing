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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly RSI(14) for overbought/oversold conditions
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_values = rsi_14_1w.fillna(50).values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w_values)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate daily ATR(14) for position sizing and stop loss
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1d = pd.Series(high_1d - low_1d)
    tr2d = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3d = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr_d = pd.concat([tr1d, tr2d, tr3d], axis=1).max(axis=1)
    atr_14_1d = tr_d.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_not_overbought = rsi_14_1w_aligned[i] < 70
        rsi_not_oversold = rsi_14_1w_aligned[i] > 30
        
        # Volatility filter: avoid low volatility periods
        volatility_filter = atr_14_1d_aligned[i] > 0
        
        # Long conditions: bullish trend + RSI not overbought + volatility
        long_condition = (price_above_ema and 
                         rsi_not_overbought and 
                         volatility_filter)
        
        # Short conditions: bearish trend + RSI not oversold + volatility
        short_condition = (price_below_ema and 
                          rsi_not_oversold and 
                          volatility_filter)
        
        # Dynamic position sizing based on volatility (inverse volatility scaling)
        if volatility_filter and atr_14_1d_aligned[i] > 0:
            # Normalize ATR to get position size (higher volatility = smaller position)
            atr_normalized = min(0.3, 0.15 * (0.01 / atr_14_1d_aligned[i]))  # Scale factor
            position_size = max(0.1, min(0.3, atr_normalized))
        else:
            position_size = 0.2
        
        if long_condition and position <= 0:
            signals[i] = position_size
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -position_size
            position = -1
        # Exit conditions: trend reversal or RSI extreme
        elif position == 1 and (not price_above_ema or rsi_14_1w_aligned[i] >= 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_14_1w_aligned[i] <= 30):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_1wRSI14_VolatilityScaled"
timeframe = "1d"
leverage = 1.0