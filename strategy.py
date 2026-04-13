#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Chaikin Money Flow (CMF) with weekly trend filter and volume confirmation.
# CMF measures institutional buying/selling pressure by combining price and volume.
# Weekly trend (EMA50) filters for direction, avoiding counter-trend trades.
# Volume confirmation ensures institutional participation. Works in both bull/bear markets
# by taking long signals only in weekly uptrend and short in downtrend.
# Target: 7-25 trades per year (30-100 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (50 + 1)
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * ema_multiplier + ema50_1w[i-1]
    
    # Daily Chaikin Money Flow (CMF) calculation
    # CMF = Sum((Close - Low) - (High - Close)) * Volume / (High - Low) over period / Sum(Volume)
    # Using 20-day period as standard
    period = 20
    mfm = np.zeros(n)  # Money Flow Multiplier
    mfv = np.zeros(n)  # Money Flow Volume
    
    # Avoid division by zero
    hl_range = high - low
    hl_range[hl_range == 0] = 1e-10
    
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # Calculate CMF using rolling sum
    cmf = np.full(n, np.nan)
    for i in range(period - 1, n):
        sum_mfv = np.sum(mfv[i - period + 1:i + 1])
        sum_volume = np.sum(volume[i - period + 1:i + 1])
        if sum_volume > 0:
            cmf[i] = sum_mfv / sum_volume
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(period - 1, n):
        # Skip if any required data is not ready
        if np.isnan(cmf[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema50_1w_aligned[i]
        cmf_val = cmf[i]
        
        if position == 0:
            # Long: CMF > 0 (buying pressure) + above weekly EMA50
            if cmf_val > 0 and price > ema_trend:
                position = 1
                signals[i] = position_size
            # Short: CMF < 0 (selling pressure) + below weekly EMA50
            elif cmf_val < 0 and price < ema_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: CMF turns negative or trend turns down
            if cmf_val <= 0 or price < ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: CMF turns positive or trend turns up
            if cmf_val >= 0 or price > ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_CMF_Trend_Volume"
timeframe = "1d"
leverage = 1.0