#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversal with 1d trend filter + volume spike
    # Long when: Williams %R(14) < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 2x avg volume
    # Short when: Williams %R(14) > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 2x avg volume
    # Exit when: Williams %R crosses above -50 (long) OR below -50 (short) OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1d EMA50 trend filter ensuring trades align with higher timeframe momentum.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d
    lookback_wr = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Williams %R conditions
        wr_oversold = williams_r_aligned[i] < -80
        wr_overbought = williams_r_aligned[i] > -20
        wr_exit_long = williams_r_aligned[i] > -50
        wr_exit_short = williams_r_aligned[i] < -50
        
        # Trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        long_entry = wr_oversold and uptrend and vol_ok and position != 1
        short_entry = wr_overbought and downtrend and vol_ok and position != -1
        
        # Exit conditions
        exit_long = wr_exit_long or (not vol_ok)
        exit_short = wr_exit_short or (not vol_ok)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williams_r_ema_volume_v1"
timeframe = "12h"
leverage = 1.0