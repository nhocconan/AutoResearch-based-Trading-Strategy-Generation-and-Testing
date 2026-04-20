#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Momentum_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily: RSI(14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[0:14] = np.nan
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        ema_val = ema_50[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(vol_ratio_val) or 
            np.isnan(atr_val) or np.isnan(ema_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when daily RSI shows momentum (not extreme)
        rsi_regime = (rsi_val > 30) and (rsi_val < 70)
        
        if position == 0:
            # Long: Price above EMA + volume confirmation + regime
            if (close_val > ema_val and   # Uptrend
                vol_ratio_val > 1.5 and    # Volume confirmation
                rsi_regime):               # Momentum regime
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA + volume confirmation + regime
            elif (close_val < ema_val and  # Downtrend
                  vol_ratio_val > 1.5 and    # Volume confirmation
                  rsi_regime):               # Momentum regime
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below EMA or volume drops
            if (close_val < ema_val) or (vol_ratio_val < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above EMA or volume drops
            if (close_val > ema_val) or (vol_ratio_val < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals