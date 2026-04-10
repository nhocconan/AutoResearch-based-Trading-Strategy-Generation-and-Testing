#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long: Williams %R(14) < -80 (oversold) + 1d EMA(20) > EMA(50) (uptrend) + 12h volume > 1.5x 20-period MA
# - Short: Williams %R(14) > -20 (overbought) + 1d EMA(20) < EMA(50) (downtrend) + 12h volume > 1.5x 20-period MA
# - Exit: Williams %R returns to -50 (mean reversion) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Williams %R identifies overextended moves; 1d EMA filter ensures trading with higher timeframe trend
# - Volume confirmation ensures institutional participation, reducing false signals in ranging markets

name = "12h_1d_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) for 12h
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 1d EMA(20) and EMA(50) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for Williams %R and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get aligned 1d data for current 12h bar (completed 1d bar)
        williams_r_current = williams_r[i]
        ema_20_current = ema_20_aligned[i]
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_12h_current = align_htf_to_ltf(prices, df_1d, volume_12h)[i]
        
        # Trend condition: EMA(20) > EMA(50) for uptrend, EMA(20) < EMA(50) for downtrend
        uptrend = ema_20_current > ema_50_current
        downtrend = ema_20_current < ema_50_current
        
        # Volume spike condition: current 12h volume > 1.5x 20-period MA
        volume_spike = volume_12h_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + uptrend + volume spike
            if (williams_r_current < -80 and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + downtrend + volume spike
            elif (williams_r_current > -20 and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 (mean reversion) or opposite signal
            if position == 1:
                if williams_r_current >= -50:  # Exit long when Williams %R crosses above -50
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_current <= -50:  # Exit short when Williams %R crosses below -50
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals