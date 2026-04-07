#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action with daily trend filter and volume confirmation
# Long when price closes above daily Donchian upper band, 12h close > daily EMA200 (uptrend), and volume > 1.5x daily average volume
# Short when price closes below daily Donchian lower band, 12h close < daily EMA200 (downtrend), and volume > 1.5x daily average volume
# Exit when trend reverses (12h close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily trend filter (EMA200) and volume confirmation for robustness
# Designed for 12h timeframe to capture multi-day moves with reduced noise
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_ema200_vol_v1"
timeframe = "12h"
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
    
    # Daily data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1d)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily EMA200 trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily average volume for confirmation
    volume_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_avg)
    
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
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_avg_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA200) or breaks below lower band
            elif close[i] < ema_200_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA200) or breaks above upper band
            elif close[i] > ema_200_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price closes above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_200_aligned[i] and
                volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action with daily trend filter and volume confirmation
# Long when price closes above daily Donchian upper band, 12h close > daily EMA200 (uptrend), and volume > 1.5x daily average volume
# Short when price closes below daily Donchian lower band, 12h close < daily EMA200 (downtrend), and volume > 1.5x daily average volume
# Exit when trend reverses (12h close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily trend filter (EMA200) and volume confirmation for robustness
# Designed for 12h timeframe to capture multi-day moves with reduced noise
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_ema200_vol_v1"
timeframe = "12h"
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
    
    # Daily data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1d)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily EMA200 trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily average volume for confirmation
    volume_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_avg)
    
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
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_avg_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA200) or breaks below lower band
            elif close[i] < ema_200_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA200) or breaks above upper band
            elif close[i] > ema_200_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price closes above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_200_aligned[i] and
                volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action with daily trend filter and volume confirmation
# Long when price closes above daily Donchian upper band, 12h close > daily EMA200 (uptrend), and volume > 1.5x daily average volume
# Short when price closes below daily Donchian lower band, 12h close < daily EMA200 (downtrend), and volume > 1.5x daily average volume
# Exit when trend reverses (12h close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily trend filter (EMA200) and volume confirmation for robustness
# Designed for 12h timeframe to capture multi-day moves with reduced noise
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_ema200_vol_v1"
timeframe = "12h"
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
    
    # Daily data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1d)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily EMA200 trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily average volume for confirmation
    volume_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_avg)
    
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
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_avg_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA200) or breaks below lower band
            elif close[i] < ema_200_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA200) or breaks above upper band
            elif close[i] > ema_200_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price closes above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_200_aligned[i] and
                volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action with daily trend filter and volume confirmation
# Long when price closes above daily Donchian upper band, 12h close > daily EMA200 (uptrend), and volume > 1.5x daily average volume
# Short when price closes below daily Donchian lower band, 12h close < daily EMA200 (downtrend), and volume > 1.5x daily average volume
# Exit when trend reverses (12h close crosses EMA200) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily trend filter (EMA200) and volume confirmation for robustness
# Designed for 12h timeframe to capture multi-day moves with reduced noise
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_ema200_vol_v1"
timeframe = "12h"
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
    
    # Daily data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian(20) channels
    high_series = pd.Series(high_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1d)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily EMA200 trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily average volume for confirmation
    volume_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_avg)
    
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
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_avg_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price below EMA200) or breaks below lower band
            elif close[i] < ema_200_aligned[i] or close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA200) or breaks above upper band
            elif close[i] > ema_200_aligned[i] or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price closes above upper band, price above EMA200 (uptrend), volume spike
            if (close[i] > upper_aligned[i] and
                close[i] > ema_200_aligned[i] and
                volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below lower band, price below EMA200 (downtrend), volume spike
            elif (close[i] < lower_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  volume[i] > 1.5 * volume_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

--- END OF FILE ---