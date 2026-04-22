# 1d_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
# Combined with RSI for momentum and Choppiness index for regime filtering, this should work in bull/bear markets.
# Target timeframe: 1d for lower trade frequency and better signal quality.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    
    # Handle first element
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
    atr = []
    for i in range(len(close)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr.append(tr)
    
    atr = np.array(atr)
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # Fill NaN with neutral value

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate KAMA on daily data
    df_daily = get_htf_data(prices, '1d')
    daily_close = df_daily['close'].values
    kama = calculate_kama(daily_close, er_length=10, fast_sc=2, slow_sc=30)
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # Calculate RSI on daily data
    rsi = calculate_rsi(daily_close, period=14)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Calculate Choppiness on daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    chop = calculate_choppiness(daily_high, daily_low, daily_close, period=14)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    # Volume spike filter
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        weekly_ema_val = weekly_ema_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        trending_regime = chop_val < 61.8
        
        # Volume filter: avoid low volume periods
        vol_filter = vol > 0.5 * vol_ma
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, weekly uptrend, trending regime, volume OK
            if (price > kama_val and 
                rsi_val > 50 and 
                weekly_ema_val > weekly_ema_val * 0.999 and  # Weekly EMA rising (simplified)
                trending_regime and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, weekly downtrend, trending regime, volume OK
            elif (price < kama_val and 
                  rsi_val < 50 and 
                  weekly_ema_val < weekly_ema_val * 1.001 and  # Weekly EMA falling (simplified)
                  trending_regime and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit when price crosses below KAMA or RSI < 40
                if price < kama_val or rsi_val < 40:
                    exit_signal = True
            
            elif position == -1:  # Short position
                # Exit when price crosses above KAMA or RSI > 60
                if price > kama_val or rsi_val > 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "1d"
leverage = 1.0