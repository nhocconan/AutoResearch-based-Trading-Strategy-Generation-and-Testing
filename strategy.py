#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1w trend filter
# - Long when price touches Camarilla L3 support with RSI(14) < 30 and 1w close > 1w EMA(50)
# - Short when price touches Camarilla H3 resistance with RSI(14) > 70 and 1w close < 1w EMA(50)
# - Uses volume confirmation: current volume > 1.5x 20-period average volume
# - Exit on opposite Camarilla level touch (H3 for longs, L3 for shorts)
# - Designed for low-frequency, high-conviction reversals in ranging markets
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h

name = "12h_1w_camarilla_rsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute daily RSI(14)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Pre-compute daily volume SMA(20)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Get previous day's OHLC for Camarilla calculation
        prev_day_idx = i // 2  # 2x 12h bars per day
        if prev_day_idx < 1 or prev_day_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        prev_high = df_1d['high'].iloc[prev_day_idx - 1]
        prev_low = df_1d['low'].iloc[prev_day_idx - 1]
        prev_close = df_1d['close'].iloc[prev_day_idx - 1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_l3 = prev_close - (range_val * 1.1 / 4)
        camarilla_h3 = prev_close + (range_val * 1.1 / 4)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w close > 1w EMA(50) for long, < for short
        uptrend = close_1w[-1] > ema_50_1w_aligned[i] if len(close_1w) > 0 else False
        downtrend = close_1w[-1] < ema_50_1w_aligned[i] if len(close_1w) > 0 else False
        
        # RSI conditions
        rsi_oversold = rsi_14_1d_aligned[i] < 30
        rsi_overbought = rsi_14_1d_aligned[i] > 70
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: price touches L3 support + oversold RSI + uptrend + volume confirmation
        if price_low <= camarilla_l3 and rsi_oversold and uptrend and vol_confirm:
            enter_long = True
        
        # Short: price touches H3 resistance + overbought RSI + downtrend + volume confirmation
        if price_high >= camarilla_h3 and rsi_overbought and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches H3 resistance
            exit_long = price_high >= camarilla_h3
        elif position == -1:
            # Exit short if price touches L3 support
            exit_short = price_low <= camarilla_l3
        
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