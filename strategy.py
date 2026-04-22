#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily CRSI (2,1,2) with weekly Bollinger Band squeeze filter and volume confirmation.
# CRSI < 10 for long, > 90 for short with price > 50-day SMA for longs and < 50-day SMA for shorts.
# Weekly Bollinger Band width < 50th percentile indicates low volatility squeeze, favoring mean reversion.
# Volume > 1.5x 20-day average confirms institutional interest.
# Designed for 1d timeframe to capture multi-day mean reversion moves in both bull and bear markets.
# Targets 10-20 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Bollinger Band squeeze filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Bollinger Bands on weekly data
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Calculate 50-period percentile of BB width for squeeze filter
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Calculate daily RSI(2) for CRSI
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI(2)
    avg_gain_2 = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss_2 = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs_2 = np.where(avg_loss_2 != 0, avg_gain_2 / avg_loss_2, 0)
    rsi_2 = 100 - (100 / (1 + rs_2))
    
    # RSI(1) on RSI(2) values
    delta_rsi2 = np.diff(rsi_2, prepend=rsi_2[0])
    gain_rsi2 = np.where(delta_rsi2 > 0, delta_rsi2, 0)
    loss_rsi2 = np.where(delta_rsi2 < 0, -delta_rsi2, 0)
    avg_gain_1 = pd.Series(gain_rsi2).ewm(alpha=1/1, adjust=False, min_periods=1).mean().values
    avg_loss_1 = pd.Series(loss_rsi2).ewm(alpha=1/1, adjust=False, min_periods=1).mean().values
    rs_1 = np.where(avg_loss_1 != 0, avg_gain_1 / avg_loss_1, 0)
    rsi_1 = 100 - (100 / (1 + rs_1))
    
    # RSI(2) on RSI(1) values to complete CRSI
    delta_rsi1 = np.diff(rsi_1, prepend=rsi_1[0])
    gain_rsi1 = np.where(delta_rsi1 > 0, delta_rsi1, 0)
    loss_rsi1 = np.where(delta_rsi1 < 0, -delta_rsi1, 0)
    avg_gain_2b = pd.Series(gain_rsi1).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss_2b = pd.Series(loss_rsi1).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs_2b = np.where(avg_loss_2b != 0, avg_gain_2b / avg_loss_2b, 0)
    rsi_2b = 100 - (100 / (1 + rs_2b))
    
    # Percentile rank of RSI(2) over 100 days for CRSI
    rsi_2_series = pd.Series(rsi_2b)
    percentrank = rsi_2_series.rolling(window=100, min_periods=100).apply(
        lambda x: (x < x[-1]).sum() / len(x) * 100 if len(x) > 0 else 50, raw=True
    ).values
    
    # CRSI = (RSI(2) + RSI(1) + Percentile Rank) / 3
    crsi = (rsi_2 + rsi_1 + percentrank) / 3
    
    # Calculate 50-day SMA for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-day average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(sma_50[i]) or 
            np.isnan(crsi[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        crsi_val = crsi[i]
        sma_val = sma_50[i]
        bb_percentile = bb_width_percentile[i]
        
        # Bollinger Band squeeze: width < 50th percentile indicates low volatility
        bb_squeeze = bb_percentile < 0.5
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: oversold CRSI + above SMA + BB squeeze + volume spike
            if crsi_val < 10 and price > sma_val and bb_squeeze and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought CRSI + below SMA + BB squeeze + volume spike
            elif crsi_val > 90 and price < sma_val and bb_squeeze and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: CRSI returns to neutral zone (40-60)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when CRSI returns to neutral or becomes overbought
                if crsi_val > 60:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when CRSI returns to neutral or becomes oversold
                if crsi_val < 40:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_CRSI_BB_Squeeze_Volume"
timeframe = "1d"
leverage = 1.0