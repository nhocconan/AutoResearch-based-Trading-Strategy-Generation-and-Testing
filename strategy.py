#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 3-bar swing high/low breakout with 1-day ATR volatility filter and volume confirmation.
# Uses swing points for precise entry timing, ATR filter to avoid low volatility whipsaws,
# and volume surge to confirm institutional participation. Designed for 6h timeframe
# to target 75-200 trades over 4 years with high win rate and controlled drawdown.
# Works in both bull and bear markets via volatility-adjusted breakouts and volume confirmation.

name = "6h_swing_breakout_atr_vol_v1"
timeframe = "6h"
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
    
    # 1-day ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]),
                    abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(close_1d), np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe (shifted by 1 day for no look-ahead)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6-hour swing points (3-bar swing high/low)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Swing high: higher than 2 bars before and after
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = high[i]
        # Swing low: lower than 2 bars before and after
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = low[i]
    
    # Forward fill swing points for breakout detection
    swing_high_ff = np.full(n, np.nan)
    swing_low_ff = np.full(n, np.nan)
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_low = swing_low[i]
        swing_high_ff[i] = last_high
        swing_low_ff[i] = last_low
    
    # Volume confirmation: 6h volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(swing_high_ff[i]) or 
            np.isnan(swing_low_ff[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Skip if ATR is zero (avoid division by zero)
        if atr_1d_aligned[i] <= 0:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2.0x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr_1d_aligned[i] > close[i] * 0.005
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below swing low or stoploss (1.5x ATR)
            stop_loss_level = entry_price - 1.5 * atr_1d_aligned[i]
            
            if (close[i] < swing_low_ff[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above swing high or stoploss
            stop_loss_level = entry_price + 1.5 * atr_1d_aligned[i]
            
            if (close[i] > swing_high_ff[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and volatility filters
            if volume_filter and vol_filter:
                # Long: breakout above swing high
                if (close[i] > swing_high_ff[i] and 
                    close[i-1] <= swing_high_ff[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below swing low
                elif (close[i] < swing_low_ff[i] and 
                      close[i-1] >= swing_low_ff[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 3-bar swing high/low breakout with 1-day ATR volatility filter and volume confirmation.
# Uses swing points for precise entry timing, ATR filter to avoid low volatility whipsaws,
# and volume surge to confirm institutional participation. Designed for 6h timeframe
# to target 75-200 trades over 4 years with high win rate and controlled drawdown.
# Works in both bull and bear markets via volatility-adjusted breakouts and volume confirmation.

name = "6h_swing_breakout_atr_vol_v1"
timeframe = "6h"
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
    
    # 1-day ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]),
                    abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(close_1d), np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe (shifted by 1 day for no look-ahead)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6-hour swing points (3-bar swing high/low)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Swing high: higher than 2 bars before and after
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            swing_high[i] = high[i]
        # Swing low: lower than 2 bars before and after
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            swing_low[i] = low[i]
    
    # Forward fill swing points for breakout detection
    swing_high_ff = np.full(n, np.nan)
    swing_low_ff = np.full(n, np.nan)
    last_high = np.nan
    last_low = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_low = swing_low[i]
        swing_high_ff[i] = last_high
        swing_low_ff[i] = last_low
    
    # Volume confirmation: 6h volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(swing_high_ff[i]) or 
            np.isnan(swing_low_ff[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Skip if ATR is zero (avoid division by zero)
        if atr_1d_aligned[i] <= 0:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2.0x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr_1d_aligned[i] > close[i] * 0.005
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below swing low or stoploss (1.5x ATR)
            stop_loss_level = entry_price - 1.5 * atr_1d_aligned[i]
            
            if (close[i] < swing_low_ff[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above swing high or stoploss
            stop_loss_level = entry_price + 1.5 * atr_1d_aligned[i]
            
            if (close[i] > swing_high_ff[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and volatility filters
            if volume_filter and vol_filter:
                # Long: breakout above swing high
                if (close[i] > swing_high_ff[i] and 
                    close[i-1] <= swing_high_ff[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below swing low
                elif (close[i] < swing_low_ff[i] and 
                      close[i-1] >= swing_low_ff[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals