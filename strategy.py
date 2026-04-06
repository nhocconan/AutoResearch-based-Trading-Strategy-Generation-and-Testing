#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Long when RSI < 30 (oversold) AND price > 4h EMA(50) (uptrend) AND volume > 1.5x 20-period average
# Short when RSI > 70 (overbought) AND price < 4h EMA(50) (downtrend) AND volume > 1.5x 20-period average
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit)
# Uses 4h trend to avoid counter-trend trades in strong trends, RSI for mean reversion entries
# Session filter (08-20 UTC) to avoid low-volume periods
# Target: 75-150 total trades over 4 years (19-38/year) for optimal 1h performance

name = "1h_rsi_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = gain_ma.values / np.maximum(loss_ma.values, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_threshold[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: RSI crosses back to neutral
        if position == 1:  # long position
            if rsi[i] >= 40:  # exit long when RSI >= 40
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 60:  # exit short when RSI <= 60
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: RSI < 30 AND price > 4h EMA(50) AND volume confirmation AND in session
            if (rsi[i] < 30 and close[i] > ema_4h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 AND price < 4h EMA(50) AND volume confirmation AND in session
            elif (rsi[i] > 70 and close[i] < ema_4h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND 4h ADX > 25 (trending) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20-period) AND 4h ADX > 25 (trending) AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 4h ADX to filter for strong trends, avoiding whipsaws in ranging markets
# Session filter (08-20 UTC) to reduce noise trades
# Target: 80-160 total trades over 4 years (20-40/year) for optimal 1h performance

name = "1h_donchian20_4h_adx_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 4h ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range and ADX
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_4h[1:] - high_4h[:-1]])
    down_move = np.concatenate([[np.nan], low_4h[:-1] - low_4h[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and DX
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.maximum(atr_4h, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.maximum(atr_4h, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx_4h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 4h ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(volume_threshold[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND 4h ADX > 25 AND volume confirmation AND in session
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                adx_4h_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower AND 4h ADX > 25 AND volume confirmation AND in session
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  adx_4h_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R mean reversion with 4h EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 4h EMA(50) (uptrend) AND volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) AND price < 4h EMA(50) (downtrend) AND volume > 1.5x 20-period average
# Exit when Williams %R crosses back to neutral (-50 for long exit, -50 for short exit)
# Uses 4h EMA trend to avoid counter-trend trades, Williams %R for mean reversion entries
# Session filter (08-20 UTC) to avoid low-volume periods
# Target: 75-150 total trades over 4 years (19-38/year) for optimal 1h performance

name = "1h_williamsr_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high.values - close) / np.maximum(highest_high.values - lowest_low.values, 1e-10)
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_threshold[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Williams %R crosses back to neutral (-50)
        if position == 1:  # long position
            if williams_r[i] >= -50:  # exit long when Williams %R >= -50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if williams_r[i] <= -50:  # exit short when Williams %R <= -50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Williams %R < -80 AND price > 4h EMA(50) AND volume confirmation AND in session
            if (williams_r[i] < -80 and close[i] > ema_4h_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: Williams %R > -20 AND price < 4h EMA(50) AND volume confirmation AND in session
            elif (williams_r[i] > -20 and close[i] < ema_4h_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals

--- End of response ---