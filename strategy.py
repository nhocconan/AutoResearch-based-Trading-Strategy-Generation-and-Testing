#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) + price > 12h EMA(50) + volume > 1.5x 20-period 4h volume SMA
# - Short when Williams %R(14) crosses below -20 (overbought) + price < 12h EMA(50) + volume > 1.5x 20-period 4h volume SMA
# - Exit: Williams %R crosses above -50 for longs or below -50 for shorts (mean reversion to midpoint)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies overextended moves ripe for mean reversion
# - 12h EMA filter ensures we trade in direction of higher timeframe trend
# - Volume confirmation avoids low-participation false signals
# - Works in bull/bear: mean reversion occurs in all regimes, trend filter improves win rate

name = "4h_12h_williamsr_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R on primary timeframe (4h) - 14 period
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          (highest_high - close) / (highest_high - lowest_low) * -100, 
                          -50.0)
    
    # Calculate 4h volume SMA for confirmation (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA for trend filter (50-period)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Williams %R signals
        wr_current = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Long: Williams %R crosses above -80 (oversold) + price > 12h EMA + volume confirmation
        long_entry = (wr_prev <= -80 and wr_current > -80 and 
                     close[i] > ema_50_12h_aligned[i] and 
                     vol_confirm)
        
        # Short: Williams %R crosses below -20 (overbought) + price < 12h EMA + volume confirmation
        short_entry = (wr_prev >= -20 and wr_current < -20 and 
                      close[i] < ema_50_12h_aligned[i] and 
                      vol_confirm)
        
        # Exit: Williams %R crosses above -50 for longs or below -50 for shorts
        exit_long = wr_prev < -50 and wr_current >= -50
        exit_short = wr_prev > -50 and wr_current <= -50
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals