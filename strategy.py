#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes breakouts; breakout direction aligned with 4h trend
# and confirmed by volume spike provides high-probability entries. Designed for 1h timeframe
# with low trade frequency (15-37/year) to minimize fee drag. Works in both bull and bear markets
# by trading breakouts in the direction of the higher timeframe trend.

name = "1h_BollingerSqueeze_Breakout_4hTrend_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 4h indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    # Calculate Bollinger Bands (20, 2) on 1h data
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + (2 * std_20)
    lower_band = ma_20 - (2 * std_20)
    bb_width = (upper_band - lower_band) / ma_20
    
    # Bollinger Band squeeze: width below 20-period mean width
    mean_bb_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < mean_bb_width
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(close[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(bb_squeeze[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Bollinger Band breakout above upper band in uptrend with volume spike and squeeze
            if close[i] > upper_band[i] and is_uptrend and volume_spike_aligned[i] and bb_squeeze[i]:
                signals[i] = 0.20
                position = 1
            # Short: Bollinger Band breakout below lower band in downtrend with volume spike and squeeze
            elif close[i] < lower_band[i] and is_downtrend and volume_spike_aligned[i] and bb_squeeze[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below middle band (mean reversion) or opposite breakout
            if close[i] < ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above middle band (mean reversion) or opposite breakout
            if close[i] > ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals