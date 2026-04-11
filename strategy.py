#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted RSI with 1d trend filter and volume confirmation
# - Uses volume-weighted RSI (VW-RSI) to capture institutional participation
# - Long when VW-RSI < 30 (oversold) + price > 1d EMA50 + volume > 1.5x average
# - Short when VW-RSI > 70 (overbought) + price < 1d EMA50 + volume > 1.5x average
# - VW-RSI reduces false signals in low-volume moves, improving signal quality
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within limits

name = "4h_1d_vwrsi_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume-weighted RSI (14-period)
    # VW-RSI = 100 - (100 / (1 + RS)), where RS = average gain / average loss
    # Weighted by volume: gain/loss * volume
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Apply volume weighting
    vol_weighted_gain = gain * volume
    vol_weighted_loss = loss * volume
    
    # Calculate smoothed averages with volume weighting
    avg_gain = pd.Series(vol_weighted_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(vol_weighted_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vwrsi = 100 - (100 / (1 + rs))
    
    # Pre-compute volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwrsi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA50 trend filter
        price_above_ema50 = price_close > ema50_1d_aligned[i]
        price_below_ema50 = price_close < ema50_1d_aligned[i]
        
        # VW-RSI conditions
        vwrsi_oversold = vwrsi[i] < 30
        vwrsi_overbought = vwrsi[i] > 70
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Oversold VW-RSI + price above 1d EMA50 + volume confirmation
        if vwrsi_oversold and price_above_ema50 and vol_confirm:
            enter_long = True
        
        # Short: Overbought VW-RSI + price below 1d EMA50 + volume confirmation
        if vwrsi_overbought and price_below_ema50 and vol_confirm:
            enter_short = True
        
        # Exit conditions: RSI returns to neutral zone or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if RSI > 50 (neutral) or price crosses below EMA50
            exit_long = (vwrsi[i] > 50) or (not price_above_ema50)
        elif position == -1:
            # Exit short if RSI < 50 (neutral) or price crosses above EMA50
            exit_short = (vwrsi[i] < 50) or (not price_below_ema50)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals