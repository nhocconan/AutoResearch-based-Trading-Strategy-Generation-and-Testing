# 1d_Weekly_Camarilla_Pivot_Squeeze_HTFTrend
# Hypothesis: 1d strategy using weekly Camarilla pivot levels (S3/R3) with 1w trend filter (EMA34) and volume confirmation.
# Weekly pivots act as strong institutional support/resistance. Trend filter ensures we trade with higher timeframe momentum.
# Volume confirms institutional interest. Designed for low trade frequency (<15/year) to minimize fee drag on 1d timeframe.
# Works in bull/bear by following weekly trend and using strong S3/R3 levels for mean-reversion or breakout.

name = "1d_Weekly_Camarilla_Pivot_Squeeze_HTFTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter and pivots
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly OHLC for Camarilla calculation (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels: S3/R3 are strongest
    # R3 = C + (H-L)*1.1
    # S3 = C - (H-L)*1.1
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1
    
    # Align weekly levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Get daily price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-day EMA (institutional interest)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34 weeks) and enough data for pivots
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA34 (uptrend) AND price breaks above R3 with volume
            if close[i] > ema_34_1w_aligned[i] and high[i] > r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA34 (downtrend) AND price breaks below S3 with volume
            elif close[i] < ema_34_1w_aligned[i] and low[i] < s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR trend turns bearish
            if low[i] < s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend turns bullish
            if high[i] > r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals