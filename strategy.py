#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return signals
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - df_4h['close'].shift())
    tr3 = np.abs(df_4h['low'] - df_4h['close'].shift())
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Close for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 4h ATR and 1d SMA to 1h timeframe
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate 1h Camarilla pivot levels from previous day
    # We'll use daily data to calculate pivots, then align to 1h
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_range = daily_high - daily_low
    
    # Camarilla levels: S1, S2, S3, S4 and R1, R2, R3, R4
    # Using formula: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    s1 = daily_close - (daily_range * 1.1 / 12)
    s2 = daily_close - (daily_range * 1.1 / 6)
    s3 = daily_close - (daily_range * 1.1 / 4)
    s4 = daily_close - (daily_range * 1.1 / 2)
    r1 = daily_close + (daily_range * 1.1 / 12)
    r2 = daily_close + (daily_range * 1.1 / 6)
    r3 = daily_close + (daily_range * 1.1 / 4)
    r4 = daily_close + (daily_range * 1.1 / 2)
    
    # Align all Camarilla levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_4h_aligned[i]) or np.isnan(sma_50_1d_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volatility filter: only trade when 4h ATR > 0.5% of price (avoid low volatility)
        volatility_filter = atr_4h_aligned[i] > (price_close * 0.005)
        
        # Trend filter: price above/below 1d SMA50
        trend_filter = price_close > sma_50_1d_aligned[i]  # for long
        trend_filter_short = price_close < sma_50_1d_aligned[i]  # for short
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > r4_aligned[i]  # Break above R4
        breakout_down = price_close < s4_aligned[i]  # Break below S4
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Break above R4 with volatility and trend filter
        if breakout_up and volatility_filter and trend_filter:
            enter_long = True
        
        # Short: Break below S4 with volatility and trend filter
        if breakout_down and volatility_filter and trend_filter_short:
            enter_short = True
        
        # Exit conditions: return to opposite S2/R2 levels (tighter stop)
        exit_long = price_close < s2_aligned[i]  # Return to S2 level
        exit_short = price_close > r2_aligned[i]  # Return to R2 level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h Camarilla breakout strategy using 4h volatility filter and 1d trend filter.
# Enters long when price breaks above daily R4 with 4h ATR > 0.5% of price and price > 1d SMA50.
# Enters short when price breaks below daily S4 with 4h ATR > 0.5% of price and price < 1d SMA50.
# Exits when price returns to S2/R2 levels for quick profit taking.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods.
# Position size fixed at 0.20 to minimize risk and allow for multiple trades.
# Target: 15-30 trades per year (60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing significant breakouts in either direction.
# The volatility filter prevents trading in choppy markets, while the trend filter ensures
# we only trade in the direction of the higher timeframe trend.
# Camarilla levels provide precise entry/exit levels based on institutional reference points.