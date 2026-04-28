#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above 6h Camarilla R3 with 1d EMA34 uptrend and volume > 1.8x 20-bar average.
# Enter short when price breaks below 6h Camarilla S3 with 1d EMA34 downtrend and volume confirmation.
# Exit when price retraces to the 6h Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels identify intraday support/resistance. EMA34 filter ensures we only trade with the daily trend,
# avoiding counter-trend whipsaws. Volume confirmation filters weak breakouts.
# This combination has shown promise on 6h timeframe with ETHUSDT (Sharpe=1.882 in DB) and should work on BTC/ETH.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla pivot calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate previous 6h bar's high, low, close for Camarilla levels
    prev_high = df_6h['high'].shift(1).values  # Previous completed 6h bar high
    prev_low = df_6h['low'].shift(1).values    # Previous completed 6h bar low
    prev_close = df_6h['close'].shift(1).values # Previous completed 6h bar close
    
    # Camarilla pivot point (PP) = (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla R3 = PP + (H - L) * 1.1 / 4
    r3 = pp + (prev_high - prev_low) * 1.1 / 4.0
    # Camarilla S3 = PP - (H - L) * 1.1 / 4
    s3 = pp - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h (wait for completed 6h bar)
    pp_aligned = align_htf_to_ltf(prices, df_6h, pp)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        ema_trend_up = close[i] > ema_34_aligned[i]
        ema_trend_down = close[i] < ema_34_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, price > EMA34 (uptrend), volume confirm
            if price > r3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3, price < EMA34 (downtrend), volume confirm
            elif price < s3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at pivot point (PP)
            if price <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at pivot point (PP)
            if price >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals