#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 12h EMA(21) > EMA(50) (uptrend)
# - Short when Williams %R(14) > -20 (overbought) AND 12h EMA(21) < EMA(50) (downtrend)
# - Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R identifies extreme price levels for mean reversion
# - 12h EMA cross ensures we trade with the higher timeframe trend
# - Works in both bull and bear markets by adapting to 12h trend direction

name = "6h_12h_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h Williams %R (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Pre-compute 12h EMAs for trend filter
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h trend: bullish when EMA21 > EMA50, bearish when EMA21 < EMA50
    ema_trend_bullish = ema_21_12h > ema_50_12h
    ema_trend_bearish = ema_21_12h < ema_50_12h
    
    # Align HTF indicators to 6h timeframe
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_bullish)
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_trend_bullish_aligned[i]) or 
            np.isnan(ema_trend_bearish_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND 12h uptrend
            if (williams_r[i] < -80 and ema_trend_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND 12h downtrend
            elif (williams_r[i] > -20 and ema_trend_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses back above -50 (long) or below -50 (short)
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals