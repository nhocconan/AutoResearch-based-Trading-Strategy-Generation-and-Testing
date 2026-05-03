#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 1d EMA34), go long when Williams %R crosses above -80 from below with volume spike.
# In bear regime (price < 1d EMA34), go short when Williams %R crosses below -20 from above with volume spike.
# Uses Williams %R(14) for mean reversion signals, 1d EMA34 for regime filter, and 6h volume spike for confirmation.
# Designed for 50-150 total trades over 4 years (12-37/year). Focus on BTC/ETH; SOL as secondary.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 periods for Williams %R
        return np.zeros(n)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align Williams %R to 6h (wait for 1d bar to complete)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        wr = williams_r_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R mean reversion signals with volume confirmation
        long_entry = False
        short_entry = False
        
        if is_bull_regime:
            # Long: Williams %R crosses above -80 from below (oversold bounce in uptrend)
            if i > 0:
                wr_prev = williams_r_aligned[i-1]
                if not np.isnan(wr_prev):
                    long_entry = (wr_prev <= -80) and (wr > -80) and vol_spike
        
        if is_bear_regime:
            # Short: Williams %R crosses below -20 from above (overbought rejection in downtrend)
            if i > 0:
                wr_prev = williams_r_aligned[i-1]
                if not np.isnan(wr_prev):
                    short_entry = (wr_prev >= -20) and (wr < -20) and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Williams %R crossing above -20 (overbought) or regime change to bear
            if wr >= -20 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Williams %R crossing below -80 (oversold) or regime change to bull
            if wr <= -80 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals