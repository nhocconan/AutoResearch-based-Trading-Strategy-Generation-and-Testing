#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1w Trend Filter and Volume Spike
# - Primary: 6h timeframe balances trade frequency and fee drag (target: 80-150 trades over 4 years)
# - HTF: 1w for trend direction (avoid counter-trend trades in strong weekly trends)
# - Long: Williams %R(14) < -80 (oversold) + weekly close > weekly EMA20 (uptrend) + volume spike
# - Short: Williams %R(14) > -20 (overbought) + weekly close < weekly EMA20 (downtrend) + volume spike
# - Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Weekly trend filter avoids fighting major moves; Williams %R captures mean reversion in ranging markets
# - Volume confirmation ensures participation and reduces false signals

name = "6h_1w_williamsr_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams %R(14) on 6h
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    
    # Calculate 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h volume moving average (20-period) for volume confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    volume_ma_20_6h_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_6h)  # Use 1w alignment for volume MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: close above/below EMA20
        weekly_uptrend = close_1w[i // (7*24)] > ema_20_1w[i // (7*24)] if i >= 7*24 else False  # Simplified: use current week's data
        weekly_downtrend = close_1w[i // (7*24)] < ema_20_1w[i // (7*24)] if i >= 7*24 else False
        
        # Better approach: use aligned weekly data
        # Get the aligned weekly close and EMA values
        # Since we can't easily get weekly index, we'll use the aligned arrays differently
        # Instead, check if the aligned weekly EMA is trending by comparing to previous aligned value
        if i > 0 and not np.isnan(ema_20_1w_aligned[i]) and not np.isnan(ema_20_1w_aligned[i-1]):
            weekly_trend_up = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
            weekly_trend_down = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        else:
            weekly_trend_up = True  # Default to allow trading initially
            weekly_trend_down = True
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        volume_spike = volume_6h[i] > 1.5 * volume_ma_20_6h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R deeply oversold + weekly uptrend + volume spike
            if (williams_r[i] < -80 and weekly_trend_up and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R deeply overbought + weekly downtrend + volume spike
            elif (williams_r[i] > -20 and weekly_trend_down and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
            if position == 1:  # Long position
                exit_condition = williams_r[i] > -50  # Exiting oversold territory
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = williams_r[i] < -50  # Exiting overbought territory
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals