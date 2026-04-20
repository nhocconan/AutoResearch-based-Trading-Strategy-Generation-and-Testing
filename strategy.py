# 1d_KAMA_EMA34_RSI_VolumeFilter_v1
# Hypothesis: 1-day KAMA trend direction combined with EMA34 trend filter and RSI momentum.
# Uses volume confirmation to avoid false signals. Designed to work in both bull and bear markets
# by only taking trades aligned with the higher timeframe (weekly) EMA34 trend.
# KAMA adapts to market noise, reducing whipsaws in choppy conditions.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Load daily data for KAMA and RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    volume_daily = df_daily['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.sum(np.abs(np.diff(close_daily)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_daily)
    for i in range(er_length, len(close_daily)):
        direction = np.abs(close_daily[i] - close_daily[i - er_length])
        volatility = np.sum(np.abs(np.diff(close_daily[i - er_length + 1:i + 1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # RSI(14) on daily
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Daily volume average (20)
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34_weekly = ema34_weekly_aligned[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma_daily = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Trend filter: only long in weekly uptrend, only short in weekly downtrend
        weekly_uptrend = price > ema34_weekly
        weekly_downtrend = price < ema34_weekly
        
        # KAMA direction: price above KAMA = bullish, below = bearish
        kama_bullish = price > kama_val
        kama_bearish = price < kama_val
        
        # RSI momentum: avoid extreme overbought/oversold
        rsi_ok = (rsi_val > 30) and (rsi_val < 70)
        
        # Volume filter: current volume > 1.3x daily average
        vol_ok = vol_current > 1.3 * vol_ma_daily
        
        if position == 0:
            # Long: price above weekly EMA34, above KAMA, RSI not overbought, with volume
            if weekly_uptrend and kama_bullish and rsi_ok and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34, below KAMA, RSI not oversold, with volume
            elif weekly_downtrend and kama_bearish and rsi_ok and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA34 OR below KAMA OR RSI overbought
            if not weekly_uptrend or not kama_bullish or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA34 OR above KAMA OR RSI oversold
            if not weekly_downtrend or not kama_bearish or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_EMA34_RSI_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0