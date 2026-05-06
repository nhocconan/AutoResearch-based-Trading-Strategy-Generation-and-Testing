#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 4h EMA50 (uptrend) AND volume > 1.5 * 24-bar avg volume
# Short when price breaks below S3 AND close < 4h EMA50 (downtrend) AND volume > 1.5 * 24-bar avg volume
# Exit when price retouches the Camarilla pivot level (mean reversion to equilibrium)
# Uses discrete sizing 0.20 to control fee drag and drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Uses 4h/1d for signal direction, 1h only for entry timing precision
# Session filter: 08-20 UTC to avoid low-liquidity periods

name = "1h_Camarilla_R3S3_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1h timeframe
    # Based on previous bar's high, low, close
    ph = np.concatenate([[high[0]], high[:-1]])  # previous high
    pl = np.concatenate([[low[0]], low[:-1]])    # previous low
    pc = np.concatenate([[close[0]], close[:-1]]) # previous close
    
    pivot = (ph + pl + pc) / 3.0
    range_ = ph - pl
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 4.0)
    s3 = pivot - (range_ * 1.1 / 4.0)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume confirmation: volume > 1.5 * 24-bar average volume (6 hours)
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * avg_volume_24)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R3 AND uptrend AND volume spike
            if close[i] > r3[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 AND downtrend AND volume spike
            elif close[i] < s3[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price retouches pivot level (mean reversion)
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price retouches pivot level (mean reversion)
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals