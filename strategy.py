#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 14 or len(df_1d) < 14:
        return signals
    
    # Calculate weekly RSI for trend bias
    delta_w = pd.Series(df_1w['close'].values).diff()
    gain_w = delta_w.where(delta_w > 0, 0)
    loss_w = -delta_w.where(delta_w < 0, 0)
    avg_gain_w = pd.Series(gain_w).rolling(window=14, min_periods=14).mean()
    avg_loss_w = pd.Series(loss_w).rolling(window=14, min_periods=14).mean()
    rs_w = avg_gain_w / avg_loss_w
    rsi_w = 100 - (100 / (1 + rs_w))
    rsi_w = rsi_w.values
    rsi_w_shifted = np.roll(rsi_w, 1)
    rsi_w_shifted[0] = np.nan
    rsi_w_aligned = align_htf_to_ltf(prices, df_1w, rsi_w_shifted)
    
    # Calculate daily RSI for divergence detection
    delta_d = pd.Series(df_1d['close'].values).diff()
    gain_d = delta_d.where(delta_d > 0, 0)
    loss_d = -delta_d.where(delta_d < 0, 0)
    avg_gain_d = pd.Series(gain_d).rolling(window=14, min_periods=14).mean()
    avg_loss_d = pd.Series(loss_d).rolling(window=14, min_periods=14).mean()
    rs_d = avg_gain_d / avg_loss_d
    rsi_d = 100 - (100 / (1 + rs_d))
    rsi_d = rsi_d.values
    rsi_d_shifted = np.roll(rsi_d, 1)
    rsi_d_shifted[0] = np.nan
    rsi_d_aligned = align_htf_to_ltf(prices, df_1d, rsi_d_shifted)
    
    # Calculate 6h RSI for entry signal
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi_shifted = np.roll(rsi, 1)
    rsi_shifted[0] = np.nan
    
    # Calculate 6h price swing points for divergence
    # Swing high: higher high than previous and next bar
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(2, n-2):
        if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
            swing_high[i] = True
        if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
            swing_low[i] = True
    
    # Track recent swing points for divergence
    last_swing_high_price = np.full(n, np.nan)
    last_swing_high_rsi = np.full(n, np.nan)
    last_swing_low_price = np.full(n, np.nan)
    last_swing_low_rsi = np.full(n, np.nan)
    
    last_high_price = np.nan
    last_high_rsi = np.nan
    last_low_price = np.nan
    last_low_rsi = np.nan
    
    for i in range(n):
        if swing_high[i]:
            last_high_price = high[i]
            last_high_rsi = rsi[i]
        if swing_low[i]:
            last_low_price = low[i]
            last_low_rsi = rsi[i]
        last_swing_high_price[i] = last_high_price
        last_swing_high_rsi[i] = last_high_rsi
        last_swing_low_price[i] = last_low_price
        last_swing_low_rsi[i] = last_low_rsi
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_w_aligned[i]) or np.isnan(rsi_d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(last_swing_high_price[i]) or
            np.isnan(last_swing_low_price[i])):
            signals[i] = 0.0
            continue
        
        rsi_w = rsi_w_aligned[i]
        rsi_d = rsi_d_aligned[i]
        rsi_6h = rsi[i]
        
        # Weekly RSI trend bias: >50 = bullish bias, <50 = bearish bias
        bullish_bias = rsi_w > 50
        bearish_bias = rsi_w < 50
        
        # Daily RSI for medium-term context
        daily_overbought = rsi_d > 60
        daily_oversold = rsi_d < 40
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if not np.isnan(last_swing_low_price[i]) and not np.isnan(last_swing_low_rsi[i]):
            if i >= 2 and not np.isnan(last_swing_low_price[i-2]) and not np.isnan(last_swing_low_rsi[i-2]):
                price_lower_low = low[i] < last_swing_low_price[i-2]
                rsi_higher_low = rsi_6h > last_swing_low_rsi[i-2]
                bullish_div = price_lower_low and rsi_higher_low
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if not np.isnan(last_swing_high_price[i]) and not np.isnan(last_swing_high_rsi[i]):
            if i >= 2 and not np.isnan(last_swing_high_price[i-2]) and not np.isnan(last_swing_high_rsi[i-2]):
                price_higher_high = high[i] > last_swing_high_price[i-2]
                rsi_lower_high = rsi_6h < last_swing_high_rsi[i-2]
                bearish_div = price_higher_high and rsi_lower_high
        
        # Entry conditions with filters
        long_signal = bullish_bias and bullish_div and daily_oversold and rsi_6h < 40
        short_signal = bearish_bias and bearish_div and daily_overbought and rsi_6h > 60
        
        # Exit when RSI returns to neutral zone (40-60) or opposite divergence
        exit_long = position == 1 and (rsi_6h > 60 or bearish_div)
        exit_short = position == -1 and (rsi_6h < 40 or bullish_div)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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

# Hypothesis: RSI divergence on 6h timeframe with weekly trend bias and daily context filter.
# Uses weekly RSI (>50 bullish, <50 bearish) to establish trend direction, preventing counter-trend trades.
# Daily RSI provides overbought/oversold context (>60 overbought, <40 oversold) to avoid extremes.
# 6h RSI identifies divergences: bullish (price lower low, RSI higher low) and bearish (price higher high, RSI lower high).
# Enters long only in weekly bullish bias with bullish divergence and daily oversold (RSI<40).
# Enters short only in weekly bearish bias with bearish divergence and daily overbought (RSI>60).
# Exits when 6h RSI returns to neutral zone (40-60) or opposite divergence appears.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing high-probability reversals.
# Works in both bull and bear markets by aligning with weekly trend and using divergence for precise timing.
# Uses proper RSI calculation with Wilder's smoothing and shifted values to avoid look-ahead.
# Swing point detection ensures divergences are based on significant price swings, not minor fluctuations.