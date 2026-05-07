# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike (>1.5x 20-period average).
# Designed for low-frequency, high-conviction trades with clear trend alignment and volume confirmation.
# Target: 15-30 trades/year per symbol to avoid excessive fee drag while maintaining edge in both bull and bear markets.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R1, S1) using previous 12h candle
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous period's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla calculations for R1/S1
        range_val = ph - pl
        camarilla_R1[i] = pc + (range_val * 1.1000 / 6)
        camarilla_S1[i] = pc - (range_val * 1.1000 / 6)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    for i in range(34, len(close_1d)):
        ema_34_1d[i] = np.mean(close_1d[i-34:i])  # Simple MA for robustness
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # Prevent overtrading (approx 2 days)
    
    start_idx = max(20, 34)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction using EMA34
        if not np.isnan(ema_34_1d_aligned[i]):
            # Need 1d close price aligned to 12h
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
            trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        else:
            trend_1d_up = False
            trend_1d_down = False
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Camarilla R1 breakout in 1d uptrend with volume spike
            if (close[i] > camarilla_R1[i] and 
                trend_1d_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Camarilla S1 breakdown in 1d downtrend with volume spike
            elif (close[i] < camarilla_S1[i] and 
                  trend_1d_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below camarilla S1 (or reverse signal)
            if close[i] < camarilla_S1[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above camarilla R1
            if close[i] > camarilla_R1[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals