#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# - Long when: BB width < 20th percentile (squeeze) AND price breaks above upper BB AND 1w close > 1w EMA50 (bullish trend) AND volume > 1.5x volume SMA20
# - Short when: BB width < 20th percentile (squeeze) AND price breaks below lower BB AND 1w close < 1w EMA50 (bearish trend) AND volume > 1.5x volume SMA20
# - Exit: opposite BB breakout or BB width expands above 50th percentile
# - Uses Bollinger squeeze to identify low volatility periods before expansion
# - Weekly trend filter ensures we trade in direction of higher timeframe momentum
# - Volume confirmation prevents false breakouts
# - Target: 15-25 trades/year to minimize fee drag while capturing explosive moves

name = "6h_1w_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute Bollinger Bands for 6h data (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Pre-compute BB width percentiles (using 50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct_20 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_width_pct_50 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.50).values
    
    # Pre-compute volume SMA for 6h data (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(bb_width_pct_20[i]) or np.isnan(bb_width_pct_50[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20th percentile (low volatility)
        squeeze = bb_width[i] < bb_width_pct_20[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_bb[i-1]  # Break above prior period's upper BB
        breakout_lower = close[i] < lower_bb[i-1]  # Break below prior period's lower BB
        
        # Weekly trend filter
        above_1w_trend = close > ema_50_1w_aligned[i]  # Price above weekly EMA50
        below_1w_trend = close < ema_50_1w_aligned[i]  # Price below weekly EMA50
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Exit conditions: opposite breakout or volatility expansion (BB width > 50th percentile)
        exit_long = breakout_lower or bb_width[i] > bb_width_pct_50[i]
        exit_short = breakout_upper or bb_width[i] > bb_width_pct_50[i]
        
        # Trading logic
        if squeeze and vol_confirm:
            # Long: BB squeeze breakout above upper BB with bullish weekly trend
            if breakout_upper and above_1w_trend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: BB squeeze breakout below lower BB with bearish weekly trend
            elif breakout_lower and below_1w_trend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No squeeze or no volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals