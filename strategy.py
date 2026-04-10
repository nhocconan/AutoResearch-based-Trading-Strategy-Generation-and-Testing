#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w EMA(21) rising AND volume > 1.5x 20-day average
# - Short when price breaks below Donchian(20) low AND 1w EMA(21) falling AND volume > 1.5x 20-day average
# - Exit when price touches opposite Donchian band (mean reversion in range) OR 1w trend reverses
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years)
# - Donchian breakouts capture strong moves; 1w EMA filter avoids counter-trend trades
# - Volume confirmation ensures breakouts have institutional participation
# - Works in both bull (breakouts continue) and bear (breakdowns continue) markets

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) upper and lower bands
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_rising = ema_21_1w > np.roll(ema_21_1w, 1)  # rising if current > previous
    ema_21_1w_falling = ema_21_1w < np.roll(ema_21_1w, 1)  # falling if current < previous
    
    # Align HTF indicators to 1d timeframe
    ema_21_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w_rising)
    ema_21_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w_falling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_21_1w_rising_aligned[i]) or 
            np.isnan(ema_21_1w_falling_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND 1w EMA rising AND volume spike
            if (close[i] > donch_high[i] and 
                ema_21_1w_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND 1w EMA falling AND volume spike
            elif (close[i] < donch_low[i] and 
                  ema_21_1w_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price touches opposite Donchian band (mean reversion)
            # 2. 1w EMA trend reverses
            exit_long = (position == 1 and 
                        (close[i] < donch_low[i] or  # touched lower band
                         not ema_21_1w_rising_aligned[i]))  # trend reversed
            exit_short = (position == -1 and 
                         (close[i] > donch_high[i] or  # touched upper band
                          not ema_21_1w_falling_aligned[i]))  # trend reversed
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals