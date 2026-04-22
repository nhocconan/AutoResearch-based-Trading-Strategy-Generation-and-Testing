#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI mean reversion with 1w EMA trend filter and volume spike
# Long when RSI < 30 (oversold) + close > 1w EMA50 (uptrend) + volume spike
# Short when RSI > 70 (overbought) + close < 1w EMA50 (downtrend) + volume spike
# Exit when RSI crosses 50 or trend reverses
# RSI captures mean reversion extremes, EMA50 filters trend direction, volume spike confirms momentum
# Designed for low trade frequency (~10-25/year) on 1d timeframe to minimize fee drain.
# Works in bull/bear by combining mean reversion with trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 14-period RSI on 1d close
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: RSI < 30 (oversold) + uptrend + volume spike
            if rsi_val < 30 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: RSI > 70 (overbought) + downtrend + volume spike
            elif rsi_val > 70 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI crosses 50 or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI crosses above 50 or trend turns down
                if rsi_val > 50 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI crosses below 50 or trend turns up
                if rsi_val < 50 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_RSI_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0