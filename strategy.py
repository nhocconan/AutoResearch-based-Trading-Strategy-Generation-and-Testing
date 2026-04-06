#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and ADX filter for trend strength
# Long when price breaks above 4h Donchian upper band, volume > 1.5x 50-period average, and ADX > 25
# Short when price breaks below 4h Donchian lower band, volume > 1.5x 50-period average, and ADX > 25
# Exit when ADX < 20 (range) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses volume confirmation to avoid false breakouts and ADX to filter ranging markets
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_volume_adx_v1"
timeframe = "4h"
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
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Volume moving average (50-period)
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ADX(14) for trend strength filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx[i]) or 
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
            # Exit: ADX weakens (<20) or breaks below lower band
            elif adx[i] < 20 or close[i] < lower_aligned[i]:
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
            # Exit: ADX weakens (<20) or breaks above upper band
            elif adx[i] < 20 or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and strong trend (ADX > 25)
            # Long: price breaks above upper band, ADX > 25 (strong trend), volume spike
            if (close[i] > upper_aligned[i] and
                adx[i] > 25 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, ADX > 25 (strong trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  adx[i] > 25 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and ADX filter for trend strength
# Long when price breaks above 4h Donchian upper band, volume > 1.5x 50-period average, and ADX > 25
# Short when price breaks below 4h Donchian lower band, volume > 1.5x 50-period average, and ADX > 25
# Exit when ADX < 20 (range) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses volume confirmation to avoid false breakouts and ADX to filter ranging markets
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_volume_adx_v1"
timeframe = "4h"
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
    
    # 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Volume moving average (50-period)
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ADX(14) for trend strength filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx[i]) or 
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
            # Exit: ADX weakens (<20) or breaks below lower band
            elif adx[i] < 20 or close[i] < lower_aligned[i]:
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
            # Exit: ADX weakens (<20) or breaks above upper band
            elif adx[i] < 20 or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and strong trend (ADX > 25)
            # Long: price breaks above upper band, ADX > 25 (strong trend), volume spike
            if (close[i] > upper_aligned[i] and
                adx[i] > 25 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, ADX > 25 (strong trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  adx[i] > 25 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals