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
    
    # Get weekly data for trend and daily data for signals
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA50 for trend filter
    ema_50_w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_w)
    
    # Daily ATR for volatility filter
    atr_14_d = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).mean().values
    atr_14_d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_d)
    
    # Daily close for momentum
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Daily RSI(14) for momentum confirmation
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_d = 100 - (100 / (1 + rs))
    rsi_14_d_values = rsi_14_d.fillna(50).values
    rsi_14_d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_d_values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for weekly EMA and daily ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_w_aligned[i]) or np.isnan(atr_14_d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(rsi_14_d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_w = ema_50_w_aligned[i]
        atr_d = atr_14_d_aligned[i]
        close_1d_val = close_1d_aligned[i]
        rsi_val = rsi_14_d_aligned[i]
        
        # Trend and volatility filters
        uptrend = price > ema_w
        downtrend = price < ema_w
        low_vol = atr_d < np.nanmedian(atr_14_d_aligned[:i+1]) * 1.5  # avoid extreme volatility
        
        if position == 0:
            # Long: uptrend, RSI not overbought, price above daily close (momentum)
            if uptrend and rsi_val < 70 and price > close_1d_val and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: downtrend, RSI not oversold, price below daily close (momentum)
            elif downtrend and rsi_val > 30 and price < close_1d_val and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or RSI overbought
            if not uptrend or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or RSI oversold
            if not downtrend or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA50_DailyRSI14_Momentum"
timeframe = "1d"
leverage = 1.0