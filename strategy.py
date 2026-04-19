#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze + weekly MACD histogram divergence + volume confirmation.
# Long when: BB width < 10th percentile (squeeze), MACD histogram crosses above zero, volume > 1.5x 20-day average
# Short when: BB width < 10th percentile, MACD histogram crosses below zero, volume > 1.5x 20-day average
# Exit when: BB width > 50th percentile (squeeze ends) or opposite MACD crossover
# Bollinger squeeze identifies low volatility breakout setups, MACD confirms momentum direction, volume validates.
# Works in bull (breakouts up) and bear (breakouts down). Target: 10-25 trades/year per symbol.
name = "1d_BB_Squeeze_MACD_Div_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for MACD
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate MACD on weekly data (12,26,9)
    ema12 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align MACD histogram to daily
    macd_hist_aligned = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    # Bollinger Bands (20,2) on daily
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_width = (upper_band - lower_band) / sma20 * 100  # percentage
    
    # Percentile rank of BB width (using 252-day lookback ~ 1 year)
    bb_width_percentile = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 20-day volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_percentile[i]) or np.isnan(macd_hist_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(macd_hist_aligned[i-1])):
            signals[i] = 0.0
            continue
        
        bb_width_pct = bb_width_percentile[i]
        macd_hist_now = macd_hist_aligned[i]
        macd_hist_prev = macd_hist_aligned[i-1]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: BB squeeze (<10th percentile), MACD hist crosses above zero, volume spike
            if (bb_width_pct < 10 and macd_hist_prev <= 0 and macd_hist_now > 0 and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze, MACD hist crosses below zero, volume spike
            elif (bb_width_pct < 10 and macd_hist_prev >= 0 and macd_hist_now < 0 and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: BB squeeze ends (>50th percentile) or MACD hist crosses below zero
            if bb_width_pct > 50 or (macd_hist_prev > 0 and macd_hist_now <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: BB squeeze ends or MACD hist crosses above zero
            if bb_width_pct > 50 or (macd_hist_prev < 0 and macd_hist_now >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals