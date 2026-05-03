#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime filter + 1d Camarilla pivot breakouts with volume confirmation
# Bollinger Band Width identifies market regime: low BW = squeeze (breakout imminent), high BW = expansion
# Trade breakouts from 1d Camarilla R3/S3 levels only during low volatility squeezes to avoid whipsaws
# Volume confirmation ensures institutional participation. Designed for 15-30 trades/year on 6h to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of volatility expansion after squeeze.

name = "6h_BBW_Squeeze_1dCamarilla_R3S3_Breakout_Volume"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use R3/S3 for fade/breakout logic
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    rango = prev_high - prev_low
    camarilla_r3 = prev_close + (rango * 1.1 / 4)
    camarilla_s3 = prev_close - (rango * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate Bollinger Band Width on 6h (20, 2)
    # Use vectorized calculation for efficiency
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + (2 * std_20)
    lower_band = ma_20 - (2 * std_20)
    bb_width = (upper_band - lower_band) / ma_20  # Normalized width
    
    # Bollinger Band Width percentile (50-period lookback) to identify squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB Width below 20th percentile (low volatility)
    squeeze_condition = bb_width_percentile < 20
    
    # Volume confirmation: 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(bb_width[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above Camarilla R3 during low volatility squeeze with volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                squeeze_condition[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below Camarilla S3 during low volatility squeeze with volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  squeeze_condition[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla R3 (mean reversion) or volatility expands significantly
            if close[i] < camarilla_r3_aligned[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla S3 (mean reversion) or volatility expands significantly
            if close[i] > camarilla_s3_aligned[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals