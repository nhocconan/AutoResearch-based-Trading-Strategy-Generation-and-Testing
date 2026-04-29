#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In ranging markets: fade extreme readings (short when > -20, long when < -80)
# In trending markets: only take signals in direction of 12h EMA50 trend
# Volume confirmation ensures breakouts have participation
# Discrete position sizing (0.25) to manage drawdown in 2022-like crashes
# Target: 12-25 trades/year on 6h (~50-100 total over 4 years) to minimize fee drag
# Works in bull markets via pullback longs in uptrend
# Works in bear markets via bounce shorts in downtrend
# Works in ranging markets via mean reversion at extremes

name = "6h_WilliamsR_MeanReversion_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for dynamic thresholds (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 50, 14)  # warmup for EMA, ATR, and Williams %R
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Determine market regime based on 12h EMA50 slope
        if i >= start_idx + 1:
            prev_ema_12h = ema_50_12h_aligned[i-1]
            ema_slope = curr_ema_12h - prev_ema_12h
            # Trending if slope magnitude > 0.1 * ATR, otherwise ranging
            is_trending = np.abs(ema_slope) > 0.1 * curr_atr
        else:
            is_trending = False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: Williams %R > -50 (mean reversion) OR stoploss
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -50 (mean reversion) OR stoploss
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            if is_trending:
                # In trending markets: only trade in direction of 12h EMA50
                if curr_ema_12h > ema_50_12h_aligned[i-1]:  # Uptrend
                    # Long on Williams %R oversold (< -80) with volume confirmation
                    if curr_williams_r < -80 and vol_spike:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend
                    # Short on Williams %R overbought (> -20) with volume confirmation
                    if curr_williams_r > -20 and vol_spike:
                        signals[i] = -0.25
                        position = -1
            else:
                # In ranging markets: mean reversion at extremes
                # Long when oversold (< -80) with volume confirmation
                if curr_williams_r < -80 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (> -20) with volume confirmation
                elif curr_williams_r > -20 and vol_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
    
    return signals