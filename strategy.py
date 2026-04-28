#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and ATR(14) stoploss.
# Enter long when price breaks above Donchian upper, 1d EMA34 trending up, and volume > 2.0x 20-bar average.
# Enter short when price breaks below Donchian lower, 1d EMA34 trending down, and volume > 2.0x 20-bar average.
# Exit when price touches Donchian midpoint OR ATR-based stoploss is hit (using highest/lowest since entry).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-120 total trades over 4 years (12-30/year).
# Donchian breakouts capture momentum; 1d EMA34 ensures higher timeframe alignment; volume spike filters noise.
# Works in bull (breakouts) and bear (breakdowns) with strict volume confirmation to avoid overtrading.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: >2.0x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    # ATR(14) for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20, 14)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirm = volume_spike[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Breakout conditions
        breakout_long = price > donchian_upper[i]
        breakout_short = price < donchian_lower[i]
        
        # Exit conditions: return to midpoint
        exit_long = price < donchian_mid[i]
        exit_short = price > donchian_mid[i]
        
        # ATR-based stoploss
        stoploss_long = False
        stoploss_short = False
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            stoploss_long = price < (highest_since_entry - 2.5 * atr[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stoploss_short = price > (lowest_since_entry + 2.5 * atr[i])
        
        # Handle entries and exits
        if breakout_long and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
            entry_price = price
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif breakout_short and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
            entry_price = price
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif position == 1 and (exit_long or stoploss_long):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        elif position == -1 and (exit_short or stoploss_short):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike (>2.0x 20-bar avg), and ATR(14) stoploss.
# Enter long when price breaks above Donchian upper, 1d EMA34 trending up, and volume > 2.0x 20-bar average.
# Enter short when price breaks below Donchian lower, 1d EMA34 trending down, and volume > 2.0x 20-bar average.
# Exit when price touches Donchian midpoint OR ATR-based stoploss is hit (using highest/lowest since entry).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-120 total trades over 4 years (12-30/year).
# Donchian breakouts capture momentum; 1d EMA34 ensures higher timeframe alignment; volume spike filters noise.
# Works in bull (breakouts) and bear (breakdowns) with strict volume confirmation to avoid overtrading.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: >2.0x 20-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    # ATR(14) for dynamic stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20, 14)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirm = volume_spike[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Breakout conditions
        breakout_long = price > donchian_upper[i]
        breakout_short = price < donchian_lower[i]
        
        # Exit conditions: return to midpoint
        exit_long = price < donchian_mid[i]
        exit_short = price > donchian_mid[i]
        
        # ATR-based stoploss
        stoploss_long = False
        stoploss_short = False
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            stoploss_long = price < (highest_since_entry - 2.5 * atr[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stoploss_short = price > (lowest_since_entry + 2.5 * atr[i])
        
        # Handle entries and exits
        if breakout_long and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
            entry_price = price
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif breakout_short and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
            entry_price = price
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif position == 1 and (exit_long or stoploss_long):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        elif position == -1 and (exit_short or stoploss_short):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals