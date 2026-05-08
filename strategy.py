# 1D_Camarilla_Pivot_Volume_Spike_Trend
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) breakouts with volume confirmation and weekly trend filter.
# - Enters long when price breaks above daily R3 with volume spike and above weekly EMA
# - Enters short when price breaks below daily S3 with volume spike and below weekly EMA
# - Exits when price crosses the daily pivot point (PP) or crosses weekly EMA in opposite direction
# - Combines Camarilla's mean-reversion levels with trend following for balanced performance
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Volume spike reduces false breakouts in low volatility periods
# - Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return in bear markets
# - Works in both bull and bear markets by following weekly trend direction
# - Camarilla levels provide institutional support/resistance with statistical edge
# - Focus on BTC and ETH as primary targets (not SOL-only)
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1D_Camarilla_Pivot_Volume_Spike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Camarilla pivot levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations: PP = (H+L+C)/3, Range = H-L
    pp = (prev_high + prev_low + prev_close) / 3.0
    rng = prev_high - prev_low
    
    # Resistance levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    r3 = prev_close + rng * 1.1 / 2.0
    s3 = prev_close - rng * 1.1 / 2.0
    
    # Volume spike detection: current volume > 2 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(pp[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        pp_val = pp[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above weekly EMA
            if (close[i] > r3[i] and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, below weekly EMA
            elif (close[i] < s3[i] and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below PP OR crosses below weekly EMA
            if (close[i] < pp_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above PP OR crosses above weekly EMA
            if (close[i] > pp_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1D_Camarilla_Pivot_Volume_Spike_Trend
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) breakouts with volume confirmation and weekly trend filter.
# - Enters long when price breaks above daily R3 with volume spike and above weekly EMA
# - Enters short when price breaks below daily S3 with volume spike and below weekly EMA
# - Exits when price crosses the daily pivot point (PP) or crosses weekly EMA in opposite direction
# - Combines Camarilla's mean-reversion levels with trend following for balanced performance
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Volume spike reduces false breakouts in low volatility periods
# - Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return in bear markets
# - Works in both bull and bear markets by following weekly trend direction
# - Camarilla levels provide institutional support/resistance with statistical edge
# - Focus on BTC and ETH as primary targets (not SOL-only)