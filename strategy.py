#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w EMA50 rising AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND 1w EMA50 falling AND volume > 1.5x 20-period average
# - Exit when price crosses Donchian(20) midline OR volume drops below average
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 10-30 trades/year on 1d timeframe (40-120 total over 4 years)
# - Donchian breakouts capture strong momentum moves
# - 1w EMA50 ensures we trade with the higher timeframe trend
# - Volume confirmation filters out weak breakouts
# - Works in both bull (trend continuation) and bear (trend acceleration) markets

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA50 trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_prev = np.roll(ema_50_1w_aligned, 1)
    ema_50_1w_prev[0] = ema_50_1w_aligned[0]  # first value
    ema_50_1w_rising = ema_50_1w_aligned > ema_50_1w_prev
    ema_50_1w_falling = ema_50_1w_aligned < ema_50_1w_prev
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND 1w EMA50 rising AND volume spike
            if (close[i] > donchian_high[i] and 
                ema_50_1w_rising[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND 1w EMA50 falling AND volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_50_1w_falling[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midline OR volume drops below average
            exit_long = (position == 1 and 
                        close[i] < donchian_mid[i])
            exit_short = (position == -1 and 
                         close[i] > donchian_mid[i])
            
            # Alternative exit: volume drops below average (weakening momentum)
            exit_vol_long = (position == 1 and 
                            volume[i] < vol_ma[i])
            exit_vol_short = (position == -1 and 
                             volume[i] < vol_ma[i])
            
            if exit_long or exit_short or exit_vol_long or exit_vol_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals