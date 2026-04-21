#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for long-term trend filter (reliable in both bull and bear)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 12h timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h RSI for momentum and mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h ATR for volatility filtering
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after warmup for EMA200
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200 = ema200_1d_aligned[i]
        rsi_val = rsi[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol = volume[i]
        price = close[i]
        
        # Volatility filter: avoid extremely low volatility environments
        vol_filter = atr_val > 0.5 * np.nanmedian(atr[max(0, i-50):i+1])
        
        # Volume confirmation: avoid low-volume false signals
        vol_filter = vol_filter and (vol > 0.7 * vol_ma_val)
        
        if position == 0:
            # Long: RSI oversold in uptrend context (bull market continuation or bear bounce)
            if rsi_val < 30 and price > ema200 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in downtrend context (bear market continuation or bull correction)
            elif rsi_val > 70 and price < ema200 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI mean reversion or trend change
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI returns to neutral or trend breaks down
                if rsi_val > 50 or price < ema200:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI returns to neutral or trend breaks up
                if rsi_val < 50 or price > ema200:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_RSI_EMA200_TrendFilter_Vol"
timeframe = "12h"
leverage = 1.0