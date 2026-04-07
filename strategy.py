#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend with daily RSI filter and hourly volume spike
# Long when KAMA trending up (price > KAMA) + RSI < 50 (pullback in uptrend) + volume > 2x 24-period hourly average
# Short when KAMA trending down (price < KAMA) + RSI > 50 (bounce in downtrend) + volume > 2x 24-period hourly average
# Exit when price crosses KAMA (reversal signal)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses KAMA for adaptive trend, RSI for pullback entry, volume spike for confirmation
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_kama_rsi_vol_v1"
timeframe = "4h"
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
    
    # 1-day data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1-hour data for volume average
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 24:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder - will compute properly below
    # Recalculate volatility properly
    volatility = np.abs(np.diff(close, prepend=close[0]))
    # For ER we need volatility over period, so we'll compute rolling sum
    change_pd = pd.Series(change)
    volatility_pd = pd.Series(volatility)
    er = change_pd.rolling(window=10, min_periods=10).sum() / (volatility_pd.rolling(window=10, min_periods=10).sum() + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1-day RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_pd = pd.Series(gain)
    loss_pd = pd.Series(loss)
    avg_gain = gain_pd.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_pd.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1-hour volume average (24-period)
    volume_1h = df_1h['volume'].values
    volume_1h_pd = pd.Series(volume_1h)
    volume_ma = volume_1h_pd.rolling(window=24, min_periods=24).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_pd = pd.Series(tr)
    atr = tr_pd.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):  # start after warmup
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses below KAMA
            elif close[i] < kama[i]:
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
            # Exit: price crosses above KAMA
            elif close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: KAMA trend with RSI pullback and volume spike
            # Trend filter: price > KAMA for long, price < KAMA for short
            kama_trend_long = close[i] > kama[i]
            kama_trend_short = close[i] < kama[i]
            # RSI filter: RSI < 50 for long (pullback), RSI > 50 for short (bounce)
            rsi_filter_long = rsi_aligned[i] < 50
            rsi_filter_short = rsi_aligned[i] > 50
            # Volume filter: volume > 2x 24-period hourly average
            volume_filter = volume[i] > 2.0 * volume_ma_aligned[i]
            
            # Long: price above KAMA + RSI < 50 + volume spike
            if kama_trend_long and rsi_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below KAMA + RSI > 50 + volume spike
            elif kama_trend_short and rsi_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour volume confirmation and daily ADX trend filter
# Long when price breaks above 4h Donchian high + volume > 1.5x 12h average volume + daily ADX > 25
# Short when price breaks below 4h Donchian low + volume > 1.5x 12h average volume + daily ADX > 25
# Exit when price crosses opposite Donchian level
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses Donchian for breakout, volume for confirmation, ADX for trend strength
# Target: 80-180 total trades over 4 years (20-45/year)

name = "4h_donchian20_12h_vol_1d_adx_v1"
timeframe = "4h"
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
    
    # 12-hour data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour volume average (20-period)
    volume_12h = df_12h['volume'].values
    volume_12h_pd = pd.Series(volume_12h)
    volume_ma = volume_12h_pd.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    
    # Calculate 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4-hour Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_pd = pd.Series(tr)
    atr = tr_pd.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
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
            # Exit: price crosses below Donchian low
            elif close[i] < lowest_low[i]:
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
            # Exit: price crosses above Donchian high
            elif close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and ADX filter
            # Volume filter: volume > 1.5x 12-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: daily ADX > 25
            trend_filter = adx_aligned[i] > 25
            
            # Long: price breaks above Donchian high + volume filter + trend filter
            if close[i] > highest_high[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + trend filter
            elif close[i] < lowest_low[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend with daily volume spike and 12-hour ADX filter
# Long when price > KAMA (uptrend) + volume > 2.0x 24-hour average + 12h ADX > 20
# Short when price < KAMA (downtrend) + volume > 2.0x 24-hour average + 12h ADX > 20
# Exit when price crosses KAMA (trend reversal)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25
# Uses KAMA for adaptive trend, volume for confirmation, ADX for trend strength
# Target: 70-150 total trades over 4 years (17-37/year)

name = "4h_kama_vol_12h_adx_v1"
timeframe = "4h"
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
    
    # 1-hour data for volume average (24-period)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 24:
        return np.zeros(n)
    
    # 12-hour data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    change_pd = pd.Series(change)
    volatility_pd = pd.Series(volatility)
    er = change_pd.rolling(window=10, min_periods=10).sum() / (volatility_pd.rolling(window=10, min_periods=10).sum() + 1e-10)
    # Smoothing constants: fast=2, slow=30
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1-hour volume average (24-period)
    volume_1h = df_1h['volume'].values
    volume_1h_pd = pd.Series(volume_1h)
    volume_ma = volume_1h_pd.rolling(window=24, min_periods=24).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1h, volume_ma)
    
    # Calculate 12-hour ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_pd = pd.Series(tr)
    atr = tr_pd.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):  # start after warmup for volume MA
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: price crosses below KAMA
            elif close[i] < kama[i]:
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
            # Exit: price crosses above KAMA
            elif close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: KAMA trend with volume spike and ADX filter
            # Trend filter: price > KAMA for long, price < KAMA for short
            kama_trend_long = close[i] > kama[i]
            kama_trend_short = close[i] < kama[i]
            # Volume filter: volume > 2.0x 24-hour average
            volume_filter = volume[i] > 2.0 * volume_ma_aligned[i]
            # Trend strength filter: 12h ADX > 20
            trend_filter = adx_aligned[i] > 20
            
            # Long: price above KAMA + volume spike + ADX > 20
            if kama_trend_long and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below KAMA + volume spike + ADX > 20
            elif kama_trend_short and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R mean reversion with 1-day volume confirmation and 1-week trend filter
# Long when Williams %R < -80 (oversold) + volume > 1.5x 24-hour average + weekly ADX > 25 (uptrend)
# Short when Williams %R > -20 (overbought) + volume > 1.5x 24-hour average + weekly ADX > 25 (downtrend)
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses Williams %R for mean reversion, volume for confirmation, weekly ADX for trend
# Target: 60-140 total trades over 4 years (15-35/year)

name = "4h_williamsr_1d_vol_1w_adx_v1"
timeframe = "4h"
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
    
    # 1-day data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    # 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1-day volume average (24-period)
    volume_1d = df_1d['volume'].values
    volume_1d_pd = pd.Series(volume_1d)
    volume_ma = volume_1d_pd.rolling(window=24, min_periods=24).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Calculate 1-week ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_pd = pd.Series(tr)
    atr = tr_pd.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(willr[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: Williams %R crosses above -50
            elif willr[i] > -50:
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
            # Exit: Williams %R crosses below -50
            elif willr[i] < -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme with volume confirmation and trend filter
            # Volume filter: volume > 1.5x 24-hour average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: weekly ADX > 25
            trend_filter = adx_aligned[i] > 25
            
            # Long: Williams %R < -80 (oversold) + volume filter + trend filter
            if willr[i] < -80 and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought) + volume filter + trend filter
            elif willr[i] > -20 and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Triple EMA crossover with 1-day volume filter and 12-hour ADX trend
# Long when EMA(8) > EMA(21) > EMA(55) + volume > 1.5x 24-hour average + 12h ADX > 25
# Short when EMA(8