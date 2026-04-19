#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Bollinger Band (20,2) squeeze breakout with volume confirmation and RSI(14) momentum filter.
# Bollinger Band squeeze indicates low volatility and impending breakout. Breakout with volume confirms strength.
# RSI(14) filters for momentum alignment to avoid false breakouts.
# Designed for 1d timeframe to capture medium-term breakouts with low frequency (target: 10-25 trades/year).
# Entry: Long when close > upper BB and BB width < 20th percentile and volume spike and RSI > 50.
# Short: when close < lower BB and BB width < 20th percentile and volume spike and RSI < 50.
# Exit: Opposite BB touch or RSI reversal (long exit if RSI < 40, short exit if RSI > 60).
# Uses tight conditions to limit trades and avoid overtrading.
name = "1d_BollingerSqueeze_Volume_RSI"
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
    
    # Weekly Bollinger Bands (20,2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Bollinger Bands: middle = SMA(20), std = 2, upper/lower = middle ± 2*std
    weekly_close_series = pd.Series(weekly_close)
    bb_middle = weekly_close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = weekly_close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower  # Band width
    
    # Align to 1d timeframe (waits for prior week close)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    
    # RSI(14) on daily close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 2.0 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Bollinger Band width percentile (20-day lookback for squeeze detection)
    bb_width_series = pd.Series(bb_width_aligned)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators (20 for BB + 20 for percentile)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        is_squeeze = bb_width_percentile[i] < 0.2
        
        if position == 0:
            # Long: break above upper BB with squeeze, volume, and bullish momentum
            if (close[i] > bb_upper_aligned[i] and 
                is_squeeze and 
                volume_spike[i] and 
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: break below lower BB with squeeze, volume, and bearish momentum
            elif (close[i] < bb_lower_aligned[i] and 
                  is_squeeze and 
                  volume_spike[i] and 
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower BB or RSI turns bearish
            if (close[i] < bb_lower_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper BB or RSI turns bullish
            if (close[i] > bb_upper_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals