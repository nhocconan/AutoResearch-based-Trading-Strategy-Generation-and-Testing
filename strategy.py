# 4h_1d_trix_volume_trend_v1
# Hypothesis: 4h TRIX (momentum) + 1d EMA200 trend filter + volume confirmation
# - TRIX: triple smoothed EMA of price, measures rate of change of a triple smoothed EMA
# - Long when TRIX > 0 (bullish momentum) + price > 1d EMA200 + volume > 1.5x 20-period average
# - Short when TRIX < 0 (bearish momentum) + price < 1d EMA200 + volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in both bull (momentum continuation with volume) and bear (momentum reversal with volume) markets
# - 1d EMA200 provides strong trend filter, reducing false signals in choppy markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_trend_v1"
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
    
    # Pre-compute TRIX on 4h data (15-period triple EMA)
    # TRIX = EMA(EMA(EMA(close), 15), 15), 15) - 1 period ago, then / previous value * 100
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.diff() / ema3.shift(1)) * 100
    trix_values = trix.values
    
    # Pre-compute 4h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_values[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # TRIX momentum
        trix_positive = trix_values[i] > 0  # Bullish momentum
        trix_negative = trix_values[i] < 0  # Bearish momentum
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA200 trend filter
        price_above_ema200 = price_close > ema200_1d_aligned[i]
        price_below_ema200 = price_close < ema200_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Positive TRIX + price above 1d EMA200 + volume confirmation
        if trix_positive and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Negative TRIX + price below 1d EMA200 + volume confirmation
        if trix_negative and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite TRIX signal or price crosses 1d EMA200
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if negative TRIX OR price crosses below 1d EMA200
            exit_long = trix_negative or (not price_above_ema200)
        elif position == -1:
            # Exit short if positive TRIX OR price crosses above 1d EMA200
            exit_short = trix_positive or (not price_below_ema200)
        
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