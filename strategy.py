#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week EMA200 trend filter and 1-day RSI(14) extremes for mean reversion.
In uptrend (price > 1w EMA200), buy when RSI < 30 (oversold). In downtrend (price < 1w EMA200), sell when RSI > 70 (overbought).
Volume must exceed 1.3x 20-day average to confirm momentum shift. Exit on trend reversal or RSI crossing 50.
Designed for 10-25 trades/year (40-100 total over 4 years) to minimize fee decay while capturing mean-reversion moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # RSI(14) on 1d closes
    delta = pd.Series(prices['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume confirmation (volume spike > 1.3x 20-day average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_200_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above weekly EMA200 + RSI oversold + volume confirmation
            if (price_close > ema_trend and 
                rsi_val < 30 and 
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly EMA200 + RSI overbought + volume confirmation
            elif (price_close < ema_trend and 
                  rsi_val > 70 and 
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR RSI crosses 50 (mean reversion complete)
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # RSI mean reversion exit
            if position == 1 and rsi_val > 50:
                exit_signal = True
            elif position == -1 and rsi_val < 50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_RSI14_1wEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0