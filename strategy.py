#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d RSI mean reversion with 1w volatility filter and volume confirmation.
Enter long when 1d RSI < 30 (oversold) and price > 1w EMA50 (uptrend filter), short when 1d RSI > 70 (overbought) and price < 1w EMA50 (downtrend filter).
Volume must exceed 1.5x 20-period average to confirm mean reversion strength.
Exit when RSI returns to neutral zone (40-60) or 1x ATR stop.
Designed for 15-30 trades/year (60-120 total over 4 years) to minimize fee fade while capturing mean reversion in ranging markets.
Works in bull markets via buying dips in uptrend and in bear markets via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for EMA50 filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1w EMA50 and 1d RSI to 6h timeframe
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_1w_50_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_filter = ema_1w_50_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: RSI oversold + price above weekly EMA50 (uptrend filter) + volume
            if (rsi_val < 30 and 
                price_close > ema_filter and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought + price below weekly EMA50 (downtrend filter) + volume
            elif (rsi_val > 70 and 
                  price_close < ema_filter and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone OR ATR-based stoploss
            exit_signal = False
            
            # RSI mean reversion exit
            if position == 1 and rsi_val > 40:
                exit_signal = True
            elif position == -1 and rsi_val < 60:
                exit_signal = True
            
            # ATR-based stoploss (1x ATR from entry approximated via recent extreme)
            if position == 1:
                # For long, stop if price drops below recent low - ATR
                recent_low = np.min(low[max(0, i-20):i+1])
                if price_close < recent_low + atr_val:  # Actually: stop when price < entry - ATR, approximate via recent low
                    exit_signal = True
            elif position == -1:
                # For short, stop if price rises above recent high + ATR
                recent_high = np.max(high[max(0, i-20):i+1])
                if price_close > recent_high - atr_val:  # Stop when price > entry + ATR
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_1dRSI_MeanReversion_1wEMA50Filter_Volume1.5x_ATR1x"
timeframe = "6h"
leverage = 1.0