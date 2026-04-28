#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme with 1w EMA50 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND close > 1w EMA50 AND volume > 1.8x 20-bar avg
# Short when Williams %R > -20 (overbought) AND close < 1w EMA50 AND volume > 1.8x 20-bar avg
# Exit when Williams %R returns to neutral zone (-50) or opposite extreme
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 15-35 trades/year on 1d.
# Works in bull markets by buying oversold dips in uptrend, works in bear by selling overbought rallies in downtrend
# Williams %R is a momentum oscillator that identifies reversal points effectively in ranging/volatile markets.

name = "1d_WilliamsR_Extreme_1wEMA50_Trend_VolumeFilter_v1"
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
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on 1d data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1w_aligned[i]
        wr = williams_r[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND close > 1w EMA50 AND volume confirmation
            if wr < -80 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND close < 1w EMA50 AND volume confirmation
            elif wr > -20 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R returns to neutral (-50) or reaches overbought
            if wr >= -50:  # Exit when momentum returns to neutral or better
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R returns to neutral (-50) or reaches oversold
            if wr <= -50:  # Exit when momentum returns to neutral or worse
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals