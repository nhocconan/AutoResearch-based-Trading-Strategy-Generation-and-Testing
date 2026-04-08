# 4h_volatility_breakout_volume_v1
# Hypothesis: 4h Bollinger Band volatility breakout with volume confirmation.
# In bull markets, volatility expands during uptrends; in bear markets, volatility spikes during breakdowns.
# Volume confirmation ensures institutional participation, reducing false breakouts.
# Target: 25-40 trades/year via volatility breakout + volume filter.

name = "4h_volatility_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20-period, 2.0 std)
    bb_period = 20
    bb_std = 2.0
    
    # Calculate SMA and std
    sma = np.zeros_like(close)
    bb_std_dev = np.zeros_like(close)
    
    for i in range(bb_period - 1, len(close)):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        bb_std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
    
    upper_band = sma + bb_std * bb_std_dev
    lower_band = sma - bb_std * bb_std_dev
    
    # Volatility expansion: current band width > 1.5x 20-period average width
    bb_width = upper_band - lower_band
    bb_width_ma = np.zeros_like(bb_width)
    for i in range(bb_period - 1, len(bb_width)):
        bb_width_ma[i] = np.mean(bb_width[i - bb_period + 1:i + 1])
    
    volatility_expansion = bb_width > 1.5 * bb_width_ma
    
    # Volume filter: current volume > 2.0x 20-period average volume
    vol_ma = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_filter = volume > 2.0 * vol_ma
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    for i in range(ema_period - 1, len(close_daily)):
        ema_daily[i] = np.mean(close_daily[i - ema_period + 1:i + 1])
    
    # Align daily EMA to 4h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(bb_period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(bb_std_dev[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit if price closes below middle band (mean reversion)
            if close[i] < sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above middle band (mean reversion)
            if close[i] > sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volatility expansion and volume
            if (close[i] > upper_band[i] and 
                volatility_expansion[i] and 
                volume_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volatility expansion and volume
            elif (close[i] < lower_band[i] and 
                  volatility_expansion[i] and 
                  volume_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals