#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h EMA(50) for trend direction (rising/falling)
# - 1h Camarilla levels (H3, L3, H4, L4) calculated from prior 4h bar
# - Long when price breaks above H3 with 4h uptrend and volume > 1.5x 20-period average
# - Short when price breaks below L3 with 4h downtrend and volume > 1.5x 20-period average
# - Exit when price returns to Camarilla pivot point (PP) or opposite breakout occurs
# - Discrete position sizing 0.20 to minimize fee churn
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 15-37 trades/year (60-150 total over 4 years) on 1h timeframe
# - Camarilla pivots work well in ranging markets; 4h trend filter adds directional bias
# - Volume confirmation reduces false breakouts
# - Designed to work in both bull (trend following) and bear (mean reversion at extremes) markets

name = "1h_4h_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Price and volume arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_4h = np.diff(ema_4h, prepend=np.nan)
    ema_rising_4h = ema_slope_4h > 0
    ema_falling_4h = ema_slope_4h < 0
    
    # Align 4h EMA trend to 1h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising_4h)
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling_4h)
    
    # Pre-compute 1d OHLC for Camarilla levels (prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # PP = (high + low + close) / 3
    rang = high_1d - low_1d
    h4 = close_1d + 1.5 * rang
    h3 = close_1d + 1.1 * rang
    l3 = close_1d - 1.1 * rang
    l4 = close_1d - 1.5 * rang
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Align 1d Camarilla levels to 1h timeframe (using prior completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for 4h EMA(50)
        # Skip if any required data is invalid or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 4h EMA rising AND volume spike
            if (close[i] > h3_aligned[i] and 
                ema_rising_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND 4h EMA falling AND volume spike
            elif (close[i] < l3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot point (PP) OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < pp_aligned[i] or close[i] < l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > pp_aligned[i] or close[i] > h3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals