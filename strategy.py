#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation.
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout direction
# determined by 12h EMA trend filter. Volume confirms breakout strength.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).
# Target: 50-150 total trades over 4 years = 12-37/year.
name = "6h_BollingerSqueeze_Breakout_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Bollinger Bands and EMA (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Bollinger Bands (20, 2) on 12h close
    close_12h = df_12h['close'].values
    bb_middle = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band width (squeeze indicator)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 12h indicators to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_12h, bb_middle)
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for BB and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20-period average (low volatility)
        squeeze = bb_width_aligned[i] < 0.02  # 2% threshold for squeeze
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: BB squeeze breakout above upper band with uptrend and volume
            if squeeze and close[i] > bb_upper_aligned[i] and close[i] > ema_34_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band with downtrend and volume
            elif squeeze and close[i] < bb_lower_aligned[i] and close[i] < ema_34_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below middle band (mean reversion)
            if close[i] < bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above middle band (mean reversion)
            if close[i] > bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals