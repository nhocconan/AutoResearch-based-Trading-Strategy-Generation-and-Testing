#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX + 14-period RSI pullback strategy with 4h EMA200 trend filter.
# Uses ADX(14) > 25 to identify trending markets, then enters on RSI pullbacks:
# - Long: ADX > 25 + RSI < 30 (oversold) + price > 4h EMA200 (uptrend)
# - Short: ADX > 25 + RSI > 70 (overbought) + price < 4h EMA200 (downtrend)
# Exits when RSI returns to neutral (40-60 range) or trend breaks.
# Designed for low trade frequency (15-30/year) by requiring strong trend + extreme RSI.
# Works in bull/bear by following 4h trend direction and only trading pullbacks within trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA200 trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 200-period EMA on 4h close
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 14-period ADX on 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 14-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        rsi_val = rsi[i]
        ema_val = ema_200_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long conditions: strong uptrend + RSI oversold
            if adx_val > 25 and price > ema_val and rsi_val < 30:
                signals[i] = 0.20
                position = 1
            # Short conditions: strong downtrend + RSI overbought
            elif adx_val > 25 and price < ema_val and rsi_val > 70:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral or trend breaks
                if rsi_val > 40 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral or trend breaks
                if rsi_val < 60 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_ADX_RSI_Pullback_4hEMA200"
timeframe = "1h"
leverage = 1.0