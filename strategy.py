#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy combining weekly KAMA trend with daily RSI mean reversion.
# Long when price pulls back to KAMA support during uptrend (RSI < 40) with volume confirmation.
# Short when price rallies to KAMA resistance during downtrend (RSI > 60) with volume confirmation.
# Exit when RSI returns to neutral (40-60) or price crosses KAMA in opposite direction.
# Uses weekly KAMA for trend, daily RSI for entry timing, volume for confirmation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Load weekly data ONCE for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly KAMA(30, 2, 30)
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=1))
        change = np.insert(change, 0, 0)
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
        # Handle first element
        volatility = np.roll(volatility, 1)
        volatility[0] = 0
        # Calculate ER and SC
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        # Initialize KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(close_1w)
    
    # Align indicators to lower timeframe
    rsi_aligned = align_htf_to_ltf(prices, prices, rsi_values)  # Already LTF
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need KAMA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(kama_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_1w_aligned[i]
        price_below_kama = close[i] < kama_1w_aligned[i]
        
        # RSI zones
        rsi_oversold = rsi_aligned[i] < 40
        rsi_overbought = rsi_aligned[i] > 60
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        if position == 0:
            # Look for mean reversion entries
            # Long: price near KAMA support in uptrend, RSI oversold
            if (price_below_kama and 
                rsi_oversold and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price near KAMA resistance in downtrend, RSI overbought
            elif (price_above_kama and 
                  rsi_overbought and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or price crosses above KAMA
            if (rsi_neutral[i] or 
                close[i] > kama_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral or price crosses below KAMA
            if (rsi_neutral[i] or 
                close[i] < kama_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyKAMA_DailyRSI_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0