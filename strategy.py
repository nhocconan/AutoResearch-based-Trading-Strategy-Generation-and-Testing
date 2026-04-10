#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) + price > 1w EMA50 (uptrend) + volume > 1.5x 12h volume SMA20
# - Short when Williams %R(14) crosses below -20 (overbought) + price < 1w EMA50 (downtrend) + volume > 1.5x 12h volume SMA20
# - Exit: Williams %R returns to -50 (mean reversion) or opposite extreme touched
# - Position sizing: 0.25 discrete level
# - Williams %R identifies overextended moves in any market regime
# - 1w EMA50 ensures we trade with the higher timeframe trend
# - Volume confirmation filters out low-conviction moves
# - Works in bull/bear: mean reversion occurs in all regimes, trend filter improves edge

name = "12h_1w_williamsr_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R on 12h timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where -20 to 0 = overbought, -80 to -100 = oversold
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume SMA for confirmation (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Williams %R signals
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        williams_cross_above_oversold = williams_r[i] > -80 and williams_r[i-1] <= -80
        williams_cross_below_overbought = williams_r[i] < -20 and williams_r[i-1] >= -20
        williams_mean_reversion = abs(williams_r[i] + 50) < 10  # near -50
        
        if position == 0:  # Flat - look for entry
            # Long: Williams %R crosses above -80 (exiting oversold) + uptrend + volume
            if williams_cross_above_oversold and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: Williams %R crosses below -20 (exiting overbought) + downtrend + volume
            elif williams_cross_below_overbought and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit: Williams %R returns to mean (-50) or reaches overbought
            if williams_mean_reversion or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit: Williams %R returns to mean (-50) or reaches oversold
            if williams_mean_reversion or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals