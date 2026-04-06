#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Donchian breakout with volume confirmation and session filter
# Long: price breaks above 4h Donchian upper band (20) + volume > 1.5x 20-bar average + 1h close > 4h EMA50 + 1d trend up (close > 1d EMA50) + session (08-20 UTC)
# Short: price breaks below 4h Donchian lower band (20) + volume > 1.5x 20-bar average + 1h close < 4h EMA50 + 1d trend down (close < 1d EMA50) + session (08-20 UTC)
# Uses 4h for signal direction (trend/structure) and 1d for trend filter, 1h only for entry timing.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled risk.
# Stoploss: 2 * ATR(14) from entry price.

name = "1h_4h_1d_donchian20_ema50_vol_sess_v1"
timeframe = "1h"
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
    open_time = prices['open_time'].values
    
    # 4h data for Donchian and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h EMA50 for additional filter (optional)
    ema50_1h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(ema50_1h[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or trend turns down
            elif close[i] < donchian_lower_4h_aligned[i] or close[i] < ema50_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or trend turns up
            elif close[i] > donchian_upper_4h_aligned[i] or close[i] > ema50_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and session filter
            # Long: price breaks above 4h Donchian upper + volume spike + 1h > 4h EMA50 + 1d trend up + session
            if (close[i] > donchian_upper_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema50_4h_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                in_session[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below 4h Donchian lower + volume spike + 1h < 4h EMA50 + 1d trend down + session
            elif (close[i] < donchian_lower_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema50_4h_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  in_session[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Donchian breakout with volume confirmation and session filter
# Long: price breaks above 4h Donchian upper band (20) + volume > 1.5x 20-bar average + 1h close > 4h EMA50 + 1d close > 1d EMA50 + session (08-20 UTC)
# Short: price breaks below 4h Donchian lower band (20) + volume > 1.5x 20-bar average + 1h close < 4h EMA50 + 1d close < 1d EMA50 + session (08-20 UTC)
# Uses 4h for signal direction (trend/structure) and 1d for trend filter, 1h only for entry timing.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled risk.
# Stoploss: 2 * ATR(14) from entry price.

name = "1h_4h_1d_donchian20_ema50_vol_sess_v1"
timeframe = "1h"
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
    open_time = prices['open_time'].values
    
    # 4h data for Donchian and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h EMA50 for additional filter (optional)
    ema50_1h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(ema50_1h[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or trend turns down
            elif close[i] < donchian_lower_4h_aligned[i] or close[i] < ema50_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or trend turns up
            elif close[i] > donchian_upper_4h_aligned[i] or close[i] > ema50_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and session filter
            # Long: price breaks above 4h Donchian upper + volume spike + 1h > 4h EMA50 + 1d trend up + session
            if (close[i] > donchian_upper_4h_aligned[i] and
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema50_4h_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                in_session[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below 4h Donchian lower + volume spike + 1h < 4h EMA50 + 1d trend down + session
            elif (close[i] < donchian_lower_4h_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema50_4h_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  in_session[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals