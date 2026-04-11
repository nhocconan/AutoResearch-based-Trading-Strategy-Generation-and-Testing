#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index + 1d EMA200 trend filter + volume confirmation
# - Elder Ray: BullPower = High - EMA13, BearPower = Low - EMA13
# - Long when BullPower > 0, BearPower rising (less negative), price > 1d EMA200, volume > 1.5x 20-period average
# - Short when BearPower < 0, BullPower falling (less positive), price < 1d EMA200, volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in both bull (strong BullPower with volume) and bear (strong BearPower with volume) markets
# - 1d EMA200 provides strong trend filter, reducing false signals in choppy markets

name = "4h_1d_elder_ray_volume_trend_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return signals
    
    # Pre-compute 1d EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute Elder Ray on 4h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Pre-compute 4h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Elder Ray signals
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        bull_rising = i > 100 and bull_power[i] > bull_power[i-1]  # BullPower increasing
        bear_falling = i > 100 and bear_power[i] < bear_power[i-1]  # BearPower decreasing (more negative)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA200 trend filter
        price_above_ema200 = price_close > ema200_1d_aligned[i]
        price_below_ema200 = price_close < ema200_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: BullPower positive AND rising + price above 1d EMA200 + volume confirmation
        if bull_positive and bull_rising and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: BearPower negative AND falling + price below 1d EMA200 + volume confirmation
        if bear_negative and bear_falling and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Elder Ray conditions or price crosses 1d EMA200
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if BearPower becomes positive OR price crosses below 1d EMA200
            exit_long = bear_power[i] > 0 or (not price_above_ema200)
        elif position == -1:
            # Exit short if BullPower becomes negative OR price crosses above 1d EMA200
            exit_short = bull_power[i] < 0 or (not price_below_ema200)
        
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