#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1-week EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold bounce) AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when Williams %R crosses below -20 (overbought rejection) AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when Williams %R crosses opposite extreme (-20 for longs, -80 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing mean reversion moves.
# Target: 30-100 trades total over 4 years (7-25/year) on 1d.
# Williams %R identifies exhaustion points; 1w EMA50 filters counter-trend moves in bear markets.
# Volume spike ensures institutional participation, reducing false signals.
# Works in bull markets (mean reversion within uptrend) and bear markets (mean reversion within downtrend).

name = "1d_WilliamsR_1wEMA50_VolumeConfirm_v1"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R on 1d data (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period) + 1  # EMA50 warmup + Williams %R warmup + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r[i-1]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1w_aligned[i]
        curr_wr = williams_r[i]
        prev_wr = williams_r[i-1]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought)
            if prev_wr <= -20 and curr_wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold)
            if prev_wr >= -80 and curr_wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R crosses above -80 (oversold bounce) AND price > 1w EMA50 AND volume confirmation
            if prev_wr <= -80 and curr_wr > -80 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 (overbought rejection) AND price < 1w EMA50 AND volume confirmation
            elif prev_wr >= -20 and curr_wr < -20 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals