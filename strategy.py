# 12h Donchian Breakout with Daily Trend and Volume Confirmation
# Hypothesis: Breakouts from daily Donchian channels on 12h timeframe with volume confirmation
# and daily EMA trend filter. This targets 50-150 trades over 4 years by requiring:
# 1) Price breakout above/below daily Donchian(20) channel
# 2) Volume > 1.5x 20-period average (12h volume confirmation)
# 3) Price above/below daily EMA50 for trend alignment
# Works in bull/bear markets via daily trend filter that prevents counter-trend trades.
# Position size: 0.25 to limit risk and reduce fee churn.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_daily_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period high/low)
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-19:i+1])
        donch_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian channels to 12h timeframe (shifted by 1 daily bar)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Daily EMA50 for trend bias
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian channels are available
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        volume_filter = not np.isnan(vol_ma) and volume[i] > vol_ma * 1.5
        
        # Trend bias: daily EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Breakout conditions
        breakout_above = close[i] > donch_high_aligned[i] and close[i-1] <= donch_high_aligned[i]
        breakout_below = close[i] < donch_low_aligned[i] and close[i-1] >= donch_low_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price re-enters Donchian channel or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] < donch_high_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price re-enters Donchian channel or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] > donch_low_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of daily trend with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if breakout_above and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif breakout_below and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
# 12h Donchian Breakout with Daily Trend and Volume Confirmation
# Hypothesis: Breakouts from daily Donchian channels on 12h timeframe with volume confirmation
# and daily EMA trend filter. This targets 50-150 trades over 4 years by requiring:
# 1) Price breakout above/below daily Donchian(20) channel
# 2) Volume > 1.5x 20-period average (12h volume confirmation)
# 3) Price above/below daily EMA50 for trend alignment
# Works in bull/bear markets via daily trend filter that prevents counter-trend trades.
# Position size: 0.25 to limit risk and reduce fee churn.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_daily_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period high/low)
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-19:i+1])
        donch_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian channels to 12h timeframe (shifted by 1 daily bar)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Daily EMA50 for trend bias
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian channels are available
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        volume_filter = not np.isnan(vol_ma) and volume[i] > vol_ma * 1.5
        
        # Trend bias: daily EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Breakout conditions
        breakout_above = close[i] > donch_high_aligned[i] and close[i-1] <= donch_high_aligned[i]
        breakout_below = close[i] < donch_low_aligned[i] and close[i-1] >= donch_low_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price re-enters Donchian channel or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] < donch_high_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price re-enters Donchian channel or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] > donch_low_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of daily trend with volume confirmation
            if volume_filter:
                # Long: breakout above Donchian high in uptrend
                if breakout_above and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below Donchian low in downtrend
                elif breakout_below and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals