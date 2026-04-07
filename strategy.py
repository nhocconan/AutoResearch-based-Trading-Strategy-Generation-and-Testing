#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour Donchian(20) breakout and 1-day EMA50 trend filter
# Long when price breaks above 4h Donchian upper band, 1d close > 1d EMA50 (uptrend), and volume > 1.8x 4h average volume
# Short when price breaks below 4h Donchian lower band, 1d close < 1d EMA50 (downtrend), and volume > 1.8x 4h average volume
# Exit when trend reverses (1d close crosses EMA50) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital) - conservative to reduce drawdown
# Uses 1d EMA50 for trend filter and 4h volume average for confirmation
# Target: 100-150 total trades over 4 years (25-38/year) - balanced for 1h timeframe
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation

name = "1h_donchian20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8-20 UTC (reduces noise trades)
    dt_index = pd.DatetimeIndex(prices['open_time'])
    hours = dt_index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h volume average for confirmation
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_4h_aligned[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA50) or breaks below lower band
            elif close[i] < ema_1d_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA50) or breaks above upper band
            elif close[i] > ema_1d_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA50 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.8 * volume_ma_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA50 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.8 * volume_ma_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour Donchian(20) breakout and 1-day EMA50 trend filter
# Long when price breaks above 4h Donchian upper band, 1d close > 1d EMA50 (uptrend), and volume > 1.8x 4h average volume
# Short when price breaks below 4h Donchian lower band, 1d close < 1d EMA50 (downtrend), and volume > 1.8x 4h average volume
# Exit when trend reverses (1d close crosses EMA50) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital) - conservative to reduce drawdown
# Uses 1d EMA50 for trend filter and 4h volume average for confirmation
# Target: 100-150 total trades over 4 years (25-38/year) - balanced for 1h timeframe
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation

name = "1h_donchian20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8-20 UTC (reduces noise trades)
    dt_index = pd.DatetimeIndex(prices['open_time'])
    hours = dt_index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h volume average for confirmation
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_4h_aligned[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA50) or breaks below lower band
            elif close[i] < ema_1d_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA50) or breaks above upper band
            elif close[i] > ema_1d_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA50 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.8 * volume_ma_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA50 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.8 * volume_ma_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour timeframe with 4-hour Donchian(20) breakout and 1-day EMA50 trend filter
# Long when price breaks above 4h Donchian upper band, 1d close > 1d EMA50 (uptrend), and volume > 1.8x 4h average volume
# Short when price breaks below 4h Donchian lower band, 1d close < 1d EMA50 (downtrend), and volume > 1.8x 4h average volume
# Exit when trend reverses (1d close crosses EMA50) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital) - conservative to reduce drawdown
# Uses 1d EMA50 for trend filter and 4h volume average for confirmation
# Target: 100-150 total trades over 4 years (25-38/year) - balanced for 1h timeframe
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation

name = "1h_donchian20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8-20 UTC (reduces noise trades)
    dt_index = pd.DatetimeIndex(prices['open_time'])
    hours = dt_index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h volume average for confirmation
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_4h_aligned[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA50) or breaks below lower band
            elif close[i] < ema_1d_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA50) or breaks above upper band
            elif close[i] > ema_1d_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above upper band, price above EMA50 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.8 * volume_ma_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, price below EMA50 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.8 * volume_ma_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

--- End of file ---