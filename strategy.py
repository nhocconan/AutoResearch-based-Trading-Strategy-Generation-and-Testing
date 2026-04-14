#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Bollinger Band squeeze breakout + volume confirmation + trend filter
# Targets: 20-50 trades/year by requiring multiple confluence factors
# Logic: Long when price breaks above upper BB (20,2) after squeeze (BBW < 20th percentile) with volume surge
#        Short when price breaks below lower BB (20,2) after squeeze with volume surge
#        Uses 12h EMA50 as trend filter to avoid counter-trend trades
# Position size: 0.25 to manage drawdown in volatile markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Bollinger Bands (20,2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze = bb_width < bb_width_percentile
    
    # Calculate 12h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / np.where(vwap_denominator > 0, vwap_denominator, np.nan)
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average (20) for volume surge detection
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned daily Bollinger data
        upper_bb_i = align_htf_to_ltf(prices, df_1d, upper_bb)[i]
        lower_bb_i = align_htf_to_ltf(prices, df_1d, lower_bb)[i]
        squeeze_i = align_htf_to_ltf(prices, df_1d, squeeze)[i]
        
        if np.isnan(upper_bb_i) or np.isnan(lower_bb_i) or np.isnan(squeeze_i) or np.isnan(vwap[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # Volume surge (2x average volume)
        volume_surge = volume[i] > 2.0 * vol_ma_20[i]
        
        # Long: BB squeeze breakout above upper BB + volume surge + above VWAP + above EMA50 (uptrend)
        if position == 0 and squeeze_i and close[i] > upper_bb_i and volume_surge and close[i] > vwap[i] and close[i] > ema_50[i]:
            position = 1
            signals[i] = position_size
        # Short: BB squeeze breakout below lower BB + volume surge + below VWAP + below EMA50 (downtrend)
        elif position == 0 and squeeze_i and close[i] < lower_bb_i and volume_surge and close[i] < vwap[i] and close[i] < ema_50[i]:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to VWAP or opposite BB break
        elif position != 0:
            if position == 1 and (close[i] < vwap[i] or close[i] < lower_bb_i):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > vwap[i] or close[i] > upper_bb_i):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_BollingerSqueeze_Breakout_VWAP_VolumeTrendFilter"
timeframe = "12h"
leverage = 1.0