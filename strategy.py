#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long: Williams %R(14) < -80 (oversold) + price > 1w EMA50 (uptrend) + 1w volume > 1.5x 20-period MA
# - Short: Williams %R(14) > -20 (overbought) + price < 1w EMA50 (downtrend) + 1w volume > 1.5x 20-period MA
# - Exit: Williams %R returns to -50 level or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Uses 1w HTF for trend and volume to filter false mean reversion signals in choppy markets
# - Williams %R captures short-term exhaustion; 1w EMA/volume ensures alignment with higher timeframe momentum
# - Works in bull/bear: mean reversion in trends with institutional participation

name = "6h_1w_williamsr_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams %R for 6h (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w volume moving average (20-period)
    volume_1w_series = pd.Series(volume_1w)
    volume_ma_20_1w = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for Williams %R and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get aligned 1w data for current 6h bar (completed 1w bar)
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Volume spike condition: current 1w volume > 1.5x 20-period MA
        volume_spike = volume_1w_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold + price above 1w EMA50 + volume spike
            if (williams_r[i] < -80 and close_price > ema_50_current and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought + price below 1w EMA50 + volume spike
            elif (williams_r[i] > -20 and close_price < ema_50_current and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 (mean reversion complete) or opposite signal
            if position == 1 and williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals