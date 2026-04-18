#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with volume confirmation and 1w trend filter.
# Works in bull (breakouts continue in trend direction) and bear (mean reversion at bands in range) via volatility expansion.
# Target: 7-25 trades/year (30-100 total over 4 years) to avoid fee drag.
name = "1d_Bollinger_Squeeze_Breakout_Volume_TrendFilter_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Bollinger Bands (20, 2) on daily
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (bb_std * bb_stddev)
    lower = sma - (bb_std * bb_stddev)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / sma
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < (0.5 * bb_width_ma)  # Squeeze when width is less than 50% of MA
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bb_width_ma[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        sma_val = sma[i]
        upper_val = upper[i]
        lower_val = lower[i]
        squeeze_val = squeeze[i]
        ema34_val = ema34_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Look for volatility expansion (end of squeeze) with volume
            if not squeeze_val and vol_filter:
                # Break above upper band in uptrend
                if close_val > upper_val and close_val > ema34_val:
                    signals[i] = 0.25
                    position = 1
                # Break below lower band in downtrend
                elif close_val < lower_val and close_val < ema34_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle (mean reversion) or volatility drops
            if close_val < sma_val or squeeze_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle (mean reversion) or volatility drops
            if close_val > sma_val or squeeze_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals