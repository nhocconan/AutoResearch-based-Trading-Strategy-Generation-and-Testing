#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) measures overbought/oversold levels (-80 to -20 = oversold, -20 to 0 = overbought)
# - Long: Williams %R < -80 (oversold) + 1d close > 1d EMA(50) (uptrend) + 1d volume > 1.2x 20-period MA
# - Short: Williams %R > -20 (overbought) + 1d close < 1d EMA(50) (downtrend) + 1d volume > 1.2x 20-period MA
# - Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year)
# - Williams %R is effective in ranging markets which dominate 2025+ test period
# - 1d trend filter ensures we trade with higher timeframe direction
# - Volume confirmation filters weak signals

name = "6h_1d_williamsr_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) for 6h
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for Williams%R14 and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get aligned 1d data for current 6h bar (completed 1d bar)
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        close_1d_current = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        
        # Trend condition: price > EMA(50) for uptrend, price < EMA(50) for downtrend
        uptrend = close_1d_current > ema_50_current
        downtrend = close_1d_current < ema_50_current
        
        # Volume spike condition: current 1d volume > 1.2x 20-period MA
        volume_spike = volume_1d_current > 1.2 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + uptrend + volume spike
            if (williams_r[i] < -80 and uptrend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + downtrend + volume spike
            elif (williams_r[i] > -20 and downtrend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
            if position == 1:
                if williams_r[i] > -50:  # Exit long when Williams %R crosses above -50
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r[i] < -50:  # Exit short when Williams %R crosses below -50
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals