#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly ATR for volatility filter
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr = np.maximum(weekly_high - weekly_low,
                    np.maximum(np.abs(weekly_high - weekly_close_prev),
                               np.abs(weekly_low - weekly_close_prev)))
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_weekly = atr_weekly / weekly_close
    
    # Align weekly ATR ratio to daily timeframe
    atr_ratio_daily = align_htf_to_ltf(prices, weekly, atr_ratio_weekly)
    
    # Calculate weekly RSI for momentum filter
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_weekly = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    rsi_daily = align_htf_to_ltf(prices, weekly, rsi_weekly)
    
    # Calculate weekly Donchian channels (breakout levels)
    donch_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe
    donch_high_daily = align_htf_to_ltf(prices, weekly, donch_high)
    donch_low_daily = align_htf_to_ltf(prices, weekly, donch_low)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_daily[i]) or np.isnan(rsi_daily[i]) or 
            np.isnan(donch_high_daily[i]) or np.isnan(donch_low_daily[i])):
            signals[i] = 0.0
            continue
        
        # Breakout strategy with volatility and momentum filters
        # Long when price breaks above weekly Donchian high in low volatility + bullish momentum
        if (close[i] > donch_high_daily[i] and 
            atr_ratio_daily[i] < 0.02 and  # Low volatility filter
            rsi_daily[i] > 50):  # Bullish momentum filter
            signals[i] = 0.25
        # Short when price breaks below weekly Donchian low in low volatility + bearish momentum
        elif (close[i] < donch_low_daily[i] and 
              atr_ratio_daily[i] < 0.02 and  # Low volatility filter
              rsi_daily[i] < 50):  # Bearish momentum filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_RSI_Volatility_Breakout"
timeframe = "1d"
leverage = 1.0