#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return signals
    
    # Calculate weekly RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Shift by 1 to use only completed weekly bars
    rsi_1w = np.roll(rsi_1w, 1)
    rsi_1w[0] = np.nan
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily RSI (14-period)
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    avg_gain_d = pd.Series(gain_d).rolling(window=14, min_periods=14).mean().values
    avg_loss_d = pd.Series(loss_d).rolling(window=14, min_periods=14).mean().values
    rs_d = np.where(avg_loss_d != 0, avg_gain_d / avg_loss_d, 0)
    rsi_d = 100 - (100 / (1 + rs_d))
    
    # Volume filter: volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_d[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        rsi_weekly = rsi_1w_aligned[i]
        rsi_daily = rsi_d[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Momentum conditions: 
        # Long when weekly RSI > 50 (bullish bias) and daily RSI crosses above 50
        # Short when weekly RSI < 50 (bearish bias) and daily RSI crosses below 50
        long_signal = volume_confirmed and (rsi_weekly > 50) and (rsi_daily > 50) and (rsi_d[i-1] <= 50)
        short_signal = volume_confirmed and (rsi_weekly < 50) and (rsi_daily < 50) and (rsi_d[i-1] >= 50)
        
        # Exit when RSI reaches opposite extreme (overbought/oversold)
        exit_long = position == 1 and (rsi_daily >= 70)
        exit_short = position == -1 and (rsi_daily <= 30)
        
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

# Hypothesis: Weekly RSI momentum with daily RSI confirmation on daily timeframe.
# Uses weekly RSI to establish long-term trend bias (above/below 50) and daily RSI for entry timing.
# Enters long when weekly RSI > 50 (bullish bias) AND daily RSI crosses above 50 with volume confirmation.
# Enters short when weekly RSI < 50 (bearish bias) AND daily RSI crosses below 50 with volume confirmation.
# Exits when daily RSI reaches overbought (>=70) for longs or oversold (<=30) for shorts.
# Works in both bull and bear markets by following the weekly momentum while using daily RSI for precise entries.
# Volume confirmation ensures institutional participation and reduces false signals.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag on daily timeframe.
# Weekly timeframe reduces noise and captures multi-week trends. RSI provides clear overbought/oversold levels.
# This strategy avoids the overtrading pitfalls of previous RSI-based strategies by requiring both weekly bias
# and daily crossover with volume confirmation, resulting in fewer but higher-quality signals.