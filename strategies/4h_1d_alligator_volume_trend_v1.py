# 4h_1d_alligator_volume_trend_v1
# Hypothesis: 4h Williams Alligator + 1d EMA trend filter + volume confirmation
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price
# - Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA200 + volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA200 + volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in both bull (trend continuation with volume) and bear (trend reversal with volume) markets
# - 1d EMA200 provides strong trend filter, reducing false signals in choppy markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_alligator_volume_trend_v1"
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
    
    # Pre-compute Williams Alligator on 4h data
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Pre-compute 4h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_median = median_price[i]
        volume_current = volume[i]
        
        # Williams Alligator alignment
        alligator_bullish = lips[i] > teeth[i] > jaw[i]  # Lips > Teeth > Jaw
        alligator_bearish = lips[i] < teeth[i] < jaw[i]  # Lips < Teeth < Jaw
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA200 trend filter
        price_above_ema200 = price_close > ema200_1d_aligned[i]
        price_below_ema200 = price_close < ema200_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish Alligator + price above 1d EMA200 + volume confirmation
        if alligator_bullish and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Bearish Alligator + price below 1d EMA200 + volume confirmation
        if alligator_bearish and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Alligator alignment or price crosses 1d EMA200
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish Alligator alignment OR price crosses below 1d EMA200
            exit_long = alligator_bearish or (not price_above_ema200)
        elif position == -1:
            # Exit short if bullish Alligator alignment OR price crosses above 1d EMA200
            exit_short = alligator_bullish or (not price_below_ema200)
        
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