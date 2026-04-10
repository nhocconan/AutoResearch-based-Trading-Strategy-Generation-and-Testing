#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA trend filter and volume confirmation
# - Long: Williams %R(14) < -80 (oversold) + 1w EMA(20) > EMA(50) (uptrend) + 1w volume > 1.2x 20-period MA
# - Short: Williams %R(14) > -20 (overbought) + 1w EMA(20) < EMA(50) (downtrend) + 1w volume > 1.2x 20-period MA
# - Exit: Williams %R returns to -50 (mean reversion) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and fee drag
# - Williams %R identifies overextended moves; 1w EMA filter ensures trading with higher timeframe trend
# - Volume confirmation ensures institutional participation, reducing false signals in ranging markets

name = "1d_1w_williamsr_meanreversion_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams %R(14) for 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 1w EMA(20) and EMA(50) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period (need at least 60 for Williams %R and EMA50)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d close
        close_price = close_1d[i]
        
        # Get aligned 1w data for current 1d bar (completed 1w bar)
        williams_r_current = williams_r[i]
        ema_20_current = ema_20_aligned[i]
        ema_50_current = ema_50_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Trend condition: EMA(20) > EMA(50) for uptrend, EMA(20) < EMA(50) for downtrend
        uptrend = ema_20_current > ema_50_current
        downtrend = ema_20_current < ema_50_current
        
        # Volume spike condition: current 1w volume > 1.2x 20-period MA
        volume_spike = volume_1w_current > 1.2 * volume_ma_current
        
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