#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe strategy using 4h trend (EMA50) and 1d momentum (RSI14) for signal direction,
# with 1h RSI(14) for entry timing and volume confirmation. Uses session filter (08-20 UTC) to reduce noise.
# Designed to work in both bull and bear markets by following higher timeframe trend and momentum.
# Target: 15-35 trades per year to minimize fee drag.

name = "1h_4hEMA50_1dRSI14_RSI14_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for RSI14 momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h EMA50 for trend direction ===
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # === 1d RSI14 for momentum ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(50).values  # Fill NaN with 50 (neutral)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # === 1h RSI14 for entry timing ===
    close = prices['close'].values
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(50).values
    
    # === 1h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    # Session filter: 08-20 UTC (pre-compute hour from index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        ema_val = ema_50_4h_aligned[i]
        rsi_1d_val = rsi_14_1d_aligned[i]
        rsi_1h_val = rsi_14_values[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(rsi_1d_val) or np.isnan(rsi_1h_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (price > EMA50), bullish momentum (RSI1d > 50), 
            #       oversold bounce (RSI1h < 30), volume confirmation
            close_val = prices['close'].iloc[i]
            if (close_val > ema_val and 
                rsi_1d_val > 50 and 
                rsi_1h_val < 30 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.20
                position = 1
            
            # Short: downtrend (price < EMA50), bearish momentum (RSI1d < 50),
            #        overbought bounce (RSI1h > 70), volume confirmation
            elif (close_val < ema_val and 
                  rsi_1d_val < 50 and 
                  rsi_1h_val > 70 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or overbought
            close_val = prices['close'].iloc[i]
            if (close_val < ema_val or rsi_1h_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal or oversold
            close_val = prices['close'].iloc[i]
            if (close_val > ema_val or rsi_1h_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals