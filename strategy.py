#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly KAMA for trend direction
    er = np.abs(close_weekly[1:] - close_weekly[:-1]) / (np.abs(close_weekly[1:] - close_weekly[:-10]) + 1e-10)
    er = np.concatenate([[0], er])
    er = np.where(np.isnan(er), 0, er)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    sc = np.where(np.isnan(sc), 0, sc)
    kama = np.zeros_like(close_weekly)
    kama[0] = close_weekly[0]
    for i in range(1, len(close_weekly)):
        kama[i] = kama[i-1] + sc[i] * (close_weekly[i] - kama[i-1])
    kama_trend = align_htf_to_ltf(prices, df_weekly, kama)
    
    # Daily KAMA for entry signal
    close_series = pd.Series(close)
    change = abs(close_series.diff())
    volatility = change.rolling(10).sum()
    er_daily = change / (volatility + 1e-10)
    er_daily = er_daily.fillna(0)
    sc_daily = (er_daily * (2/2 - 2/30) + 2/30) ** 2
    sc_daily = sc_daily.fillna(0)
    kama_daily = np.zeros(n)
    kama_daily[0] = close[0]
    for i in range(1, n):
        kama_daily[i] = kama_daily[i-1] + sc_daily[i] * (close[i] - kama_daily[i-1])
    
    # Daily RSI for overbought/oversold
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness index for regime detection
    atr1 = high - low
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    tr[0] = atr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(kama_trend[i]) or np.isnan(kama_daily[i]) or np.isnan(rsi[i]) or \
           np.isnan(chop[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_t = kama_trend[i]
        kama_d = kama_daily[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        trending_market = chop_val < 38.2  # Trending regime
        ranging_market = chop_val > 61.8   # Ranging regime
        
        # Determine market regime
        if trending_market:
            regime = "trend"
        elif ranging_market:
            regime = "range"
        else:
            regime = "transition"
        
        if position == 0:
            # Long conditions
            long_condition = False
            if regime == "trend":
                # Trend following: price above KAMA and weekly trend up
                long_condition = price > kama_d and close_weekly[-1] > kama_trend[i]
            elif regime == "range":
                # Mean reversion: RSI oversold in range
                long_condition = rsi_val < 30
            
            if long_condition and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # Short conditions
            short_condition = False
            if regime == "trend":
                # Trend following: price below KAMA and weekly trend down
                short_condition = price < kama_d and close_weekly[-1] < kama_trend[i]
            elif regime == "range":
                # Mean reversion: RSI overbought in range
                short_condition = rsi_val > 70
            
            if short_condition and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long
            exit_condition = False
            if regime == "trend":
                exit_condition = price < kama_d
            elif regime == "range":
                exit_condition = rsi_val > 70
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short
            exit_condition = False
            if regime == "trend":
                exit_condition = price > kama_d
            elif regime == "range":
                exit_condition = rsi_val < 30
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals