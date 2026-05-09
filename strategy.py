#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume spike.
# Uses Bollinger Bands (20,2) squeeze detection (<20th percentile width) followed by breakout.
# Long when price breaks above upper band with 4h uptrend and volume spike.
# Short when price breaks below lower band with 4h downtrend and volume spike.
# Designed to work in both bull (follow 4h uptrend) and bear (follow 4h downtrend) markets.
# Target: 15-37 trades/year to avoid fee drag.
name = "1h_Bollinger_Squeeze_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on 1h
    bb_period = 20
    bb_mult = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper = sma + bb_mult * std
    lower = sma - bb_mult * std
    bb_width = upper - lower
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20)
    squeeze = bb_width < bb_width_percentile.values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema20[i]) or 
            np.isnan(sma.iloc[i]) or np.isnan(std.iloc[i]) or np.isnan(bb_width_percentile.iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper_val = upper.iloc[i]
        lower_val = lower.iloc[i]
        squeeze_val = squeeze.iloc[i]
        
        if position == 0:
            # Enter long: BB squeeze breakout above upper band + 4h uptrend + volume spike
            if squeeze_val and price > upper_val and price > ema_50_4h_aligned[i] and vol_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: BB squeeze breakout below lower band + 4h downtrend + volume spike
            elif squeeze_val and price < lower_val and price < ema_50_4h_aligned[i] and vol_confirm[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns below middle Bollinger Band or trend reverses
            if price < sma.iloc[i] or price < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns above middle Bollinger Band or trend reverses
            if price > sma.iloc[i] or price > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals