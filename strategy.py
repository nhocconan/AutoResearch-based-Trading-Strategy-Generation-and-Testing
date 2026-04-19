#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with volume confirmation and weekly trend filter.
# Uses weekly EMA20 as trend filter (bullish when price > weekly EMA20).
# Long when price breaks above upper BB(20,2) with volume > 1.5x 20-day average.
# Short when price breaks below lower BB(20,2) with volume confirmation and bearish weekly trend.
# Weekly trend filter reduces false breakouts in choppy markets.
# Target: 15-25 trades/year per symbol, focusing on high-conviction breakouts.
name = "1d_BollingerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Bollinger Bands (20, 2) on daily
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20)  # Ensure BB and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_20_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma_val = sma_20[i]
        upper = upper_band[i]
        lower = lower_band[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        weekly_ema = ema_20_weekly_aligned[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper BB, bullish weekly trend, volume confirmation
            if price > upper and price > weekly_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB, bearish weekly trend, volume confirmation
            elif price < lower and price < weekly_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below middle Bollinger Band (SMA20)
            if price < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above middle Bollinger Band (SMA20)
            if price > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals