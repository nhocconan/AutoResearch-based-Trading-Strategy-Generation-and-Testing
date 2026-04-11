# 1d_1w_camarilla_breakout_volume_v3
# Hypothesis: Daily Camarilla breakout with weekly trend filter and volume confirmation.
# Uses weekly RSI to determine trend direction and only trades breakouts in the direction of weekly trend.
# Reduces trades by requiring alignment between daily breakout and weekly trend.
# Weekly trend filter helps avoid counter-trend trades in choppy markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 15-25 trades per year to minimize fee drag while capturing significant moves.
# Works in bull markets (long breakouts in uptrend) and bear markets (short breakdowns in downtrend).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v3"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate weekly RSI for trend filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate Camarilla pivot levels from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # Resistance levels: R1 = close + (range * 1.1/12), R2 = close + (range * 1.1/6), R3 = close + (range * 1.1/4), R4 = close + (range * 1.1/2)
    # Support levels: S1 = close - (range * 1.1/12), S2 = close - (range * 1.1/6), S3 = close - (range * 1.1/4), S4 = close - (range * 1.1/2)
    daily_range = high_1d - low_1d
    
    # Key levels for breakout: R4 (resistance) and S4 (support)
    r4 = close_1d + (daily_range * 1.1 / 2)
    s4 = close_1d - (daily_range * 1.1 / 2)
    
    # Exit levels: R3 and S3
    r3 = close_1d + (daily_range * 1.1 / 4)
    s3 = close_1d - (daily_range * 1.1 / 4)
    
    # Volume confirmation: daily volume > 1.5x 20-period average (moderate to balance signal quality and frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly and daily data to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Weekly trend filter: RSI > 50 = uptrend, RSI < 50 = downtrend
        weekly_uptrend = rsi_1w_aligned[i] > 50
        weekly_downtrend = rsi_1w_aligned[i] < 50
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > r4_aligned[i]   # Break above R4
        breakout_down = price_close < s4_aligned[i] # Break below S4
        
        # Entry conditions: only trade in direction of weekly trend
        enter_long = False
        enter_short = False
        
        # Long: Break above R4 with volume confirmation AND weekly uptrend
        if breakout_up and vol_confirm and weekly_uptrend:
            enter_long = True
        
        # Short: Break below S4 with volume confirmation AND weekly downtrend
        if breakout_down and vol_confirm and weekly_downtrend:
            enter_short = True
        
        # Exit conditions: return to opposite S3/R3 levels
        exit_long = price_close < s3_aligned[i]   # Return to S3 level
        exit_short = price_close > r3_aligned[i]  # Return to R3 level
        
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