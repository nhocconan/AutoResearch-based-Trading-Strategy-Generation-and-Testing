#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with 12-hour EMA trend filter and 1-day volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and rising + price > 12h EMA(50) + volume > 1.5x 20-period average
# Short when Bear Power > 0 and rising + price < 12h EMA(50) + volume > 1.5x 20-period average
# Exit when power signal reverses or price crosses 12h EMA(50) in opposite direction
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_elder_ray_12h_ema_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(13) for Elder Ray (using close prices)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = ema_13 - low   # EMA(13) - Low
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power becomes positive and rising OR price below 12h EMA(50)
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power becomes positive and rising OR price above 12h EMA(50)
            elif bull_power[i] > 0 and bull_power[i] > bull_power[i-1] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with trend and volume filters
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: Bull Power > 0 and rising + price > 12h EMA(50) + volume filter
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and close[i] > ema_50_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power > 0 and rising + price < 12h EMA(50) + volume filter
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and close[i] < ema_50_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with 12-hour EMA trend filter and 1-day volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and rising + price > 12h EMA(50) + volume > 1.5x 20-period average
# Short when Bear Power > 0 and rising + price < 12h EMA(50) + volume > 1.5x 20-period average
# Exit when power signal reverses or price crosses 12h EMA(50) in opposite direction
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_elder_ray_12h_ema_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA(13) for Elder Ray (using close prices)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA(13)
    bear_power = ema_13 - low   # EMA(13) - Low
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power becomes positive and rising OR price below 12h EMA(50)
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power becomes positive and rising OR price above 12h EMA(50)
            elif bull_power[i] > 0 and bull_power[i] > bull_power[i-1] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray signals with trend and volume filters
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: Bull Power > 0 and rising + price > 12h EMA(50) + volume filter
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and close[i] > ema_50_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power > 0 and rising + price < 12h EMA(50) + volume filter
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and close[i] < ema_50_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals