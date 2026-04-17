# 6H_1D_WPIVOT_TREND_CONFIRMED
# Hypothesis: Weekly pivot points (calculated from Sunday open, weekly high/low/close) act as key support/resistance on the 6h timeframe.
# In trending markets (6h price above/below 1d EMA50), price breaks through these levels with momentum.
# In ranging markets, price respects these levels as reversal points.
# Uses volume confirmation (1.5x 20-period average) to filter false breakouts.
# Targets 15-35 trades/year on 6f timeframe by requiring confluence of weekly pivot, trend, and volume.
# Works in bull markets via breakout continuation and bear markets via mean reversion from weekly extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (weekly = 7 * 24h = 168h, but we use 1w from data)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot and support/resistance levels
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    wopen = df_1w['open'].values  # for alternative pivot if needed
    
    # Standard weekly pivot: (H + L + C) / 3
    wpivot = (whigh + wlow + wclose) / 3
    wrange = whigh - wlow
    
    # Weekly support/resistance levels
    w_r1 = 2 * wpivot - wlow
    w_s1 = 2 * wpivot - whigh
    w_r2 = wpivot + wrange
    w_s2 = wpivot - wrange
    
    # Calculate 1d EMA50 for trend filter (6h chart uses daily trend)
    df_1d = get_htf_data(prices, '1d')
    dclose = df_1d['close'].values
    ema_50 = pd.Series(dclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly levels to 6h timeframe (waits for weekly bar to close)
    wpivot_6h = align_htf_to_ltf(prices, df_1w, wpivot)
    w_r1_6h = align_htf_to_ltf(prices, df_1w, w_r1)
    w_s1_6h = align_htf_to_ltf(prices, df_1w, w_s1)
    w_r2_6h = align_htf_to_ltf(prices, df_1w, w_r2)
    w_s2_6h = align_htf_to_ltf(prices, df_1w, w_s2)
    
    # Align daily EMA50 to 6h
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators (50 for EMA + buffer)
    
    for i in range(start_idx, n):
        if (np.isnan(wpivot_6h[i]) or np.isnan(w_r1_6h[i]) or np.isnan(w_s1_6h[i]) or
            np.isnan(w_r2_6h[i]) or np.isnan(w_s2_6h[i]) or np.isnan(ema_50_6h[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        # Determine market regime: trending if price > EMA50 (bull) or < EMA50 (bear)
        is_bull_trend = price > ema_50_6h[i]
        is_bear_trend = price < ema_50_6h[i]
        
        if position == 0:
            # Long conditions:
            # 1. Breakout above weekly R1 with volume spike AND bullish trend
            # 2. Mean reversion from weekly S1 with volume spike AND bearish trend (reversal up)
            if ((price > w_r1_6h[i] and vol > 1.5 * vol_ma and is_bull_trend) or
                (price < w_s1_6h[i] and vol > 1.5 * vol_ma and is_bear_trend)):
                signals[i] = 0.25
                position = 1
            
            # Short conditions:
            # 1. Breakdown below weekly S1 with volume spike AND bearish trend
            # 2. Mean reversion from weekly R1 with volume spike AND bullish trend (reversal down)
            elif ((price < w_s1_6h[i] and vol > 1.5 * vol_ma and is_bear_trend) or
                  (price > w_r1_6h[i] and vol > 1.5 * vol_ma and is_bull_trend)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot or hits weekly R2 (take profit)
            if price < wpivot_6h[i] or price > w_r2_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot or hits weekly S2 (take profit)
            if price > wpivot_6h[i] or price < w_s2_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6H_1D_WPIVOT_TREND_CONFIRMED"
timeframe = "6h"
leverage = 1.0