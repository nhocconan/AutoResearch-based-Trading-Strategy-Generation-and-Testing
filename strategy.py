#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
# Goes long when price breaks above Donchian upper band with volume > 1.5x average and ADX > 20.
# Goes short when price breaks below Donchian lower band with volume > 1.5x average and ADX > 20.
# Uses 1d EMA(50) trend filter to align with higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_donchian20_vol_adx_trend_v1"
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
    
    # Donchian(20) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ADX(14) for trend filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr * 100
    minus_di = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_strong[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or 1d EMA turns down
            elif close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or 1d EMA turns up
            elif close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and ADX filter
            if vol_strong[i] and adx[i] > 20:
                # Long breakout: price breaks above Donchian upper
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian lower
                elif close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX(14) trend strength with Donchian(20) breakout and volume confirmation.
# Goes long when ADX > 25 (strong trend) and price breaks above Donchian upper with volume > 1.3x average.
# Goes short when ADX > 25 and price breaks below Donchian lower with volume > 1.3x average.
# Uses 1d EMA(50) to filter trades in direction of higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_adx25_donchian20_vol_trend_v1"
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
    
    # Donchian(20) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)  # Volume above average
    
    # ADX(14) for trend strength
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr * 100
    minus_di = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or 1d EMA turns down
            elif close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or 1d EMA turns up
            elif close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with ADX trend filter and volume
            if adx[i] > 25 and vol_filter[i]:
                # Long breakout: price breaks above Donchian upper
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian lower
                elif close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-weighted average price (VWAP) deviation with Bollinger Bands squeeze.
# Goes long when price closes below VWAP - 1.5*std and BB width < 50th percentile (squeeze).
# Goes short when price closes above VWAP + 1.5*std and BB width < 50th percentile.
# Uses 1d ADX > 20 to ensure trending environment for mean reversion to work.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_vwap_bb_squeeze_v1"
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
    
    # VWAP calculation
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # Bollinger Bands(20, 2)
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2.0 * bb_std
    bb_lower = bb_ma - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # BB width percentile for squeeze detection (lookback 50 periods)
    bb_width_pct = np.full(n, np.nan)
    for i in range(50, n):
        bb_width_pct[i] = pd.Series(bb_width[i-49:i+1]).rank(pct=True).iloc[-1] * 100
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX(14) calculation for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d * 100
    minus_di = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(vwap[i]) or np.isnan(bb_width_pct[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            tr = high[i] - low[i]
            tr = max(tr, abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i]
            atr_est = tr  # Simplified ATR for stop
            if close[i] < entry_price - 2.0 * atr_est:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above VWAP or BB width expands
            elif close[i] > vwap[i] or bb_width_pct[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            tr = high[i] - low[i]
            tr = max(tr, abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i]
            atr_est = tr  # Simplified ATR for stop
            if close[i] > entry_price + 2.0 * atr_est:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below VWAP or BB width expands
            elif close[i] < vwap[i] or bb_width_pct[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries in squeeze
            if adx_1d_aligned[i] > 20:  # Trending environment
                # Long when price is below VWAP - 1.5*std and in squeeze
                if close[i] < (vwap[i] - 1.5 * bb_std[i]) and bb_width_pct[i] < 40:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price is above VWAP + 1.5*std and in squeeze
                elif close[i] > (vwap[i] + 1.5 * bb_std[i]) and bb_width_pct[i] < 40:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
# Goes long when price breaks above Donchian upper band with volume > 1.5x average and ADX > 20.
# Goes short when price breaks below Donchian lower band with volume > 1.5x average and ADX > 20.
# Uses 1d EMA(50) trend filter to align with higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_donchian20_vol_adx_trend_v1"
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
    
    # Donchian(20) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ADX(14) for trend filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr * 100
    minus_di = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_strong[i]) or 
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or 1d EMA turns down
            elif close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or 1d EMA turns up
            elif close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and ADX filter
            if vol_strong[i] and adx[i] > 20:
                # Long breakout: price breaks above Donchian upper
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian lower
                elif close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout.
# Goes long when CHOP > 61.8 (ranging market) and price breaks above Donchian upper with volume > 1.3x average.
# Goes short when CHOP > 61.8 and price breaks below Donchian lower with volume > 1.3x average.
# Uses 1d EMA(50) to filter trades in direction of higher timeframe trend.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "4h_chop_donchian20_vol_trend_v1"
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
    
    # Donchian(20) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)  # Volume above average
    
    # Choppiness Index(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower or 1d EMA turns down
            elif close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper or 1d EMA turns up
            elif close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (high chop) with volume filter
            if chop[i] > 61.8 and vol_filter[i]:
                # Long breakout: price breaks above Donchian upper
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian lower
                elif close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

--- Previous strategy had 0 trades. Making ADX trend filter stricter and adding volume confirmation. ---
  #16475 [discard] 6h_camarilla1d_vol_trend_v1 | Sharpe=-0.370 (308 tr/sym) ← neg_sharpe
  #16493 [discard] 4h_donchian20_kama12_vol_v1 | Sharpe=0.000 ← too_few_trades(0tr)
  #16497 [discard] 4h_donchian20_vol_adx_trend_v1 | Sharpe=0.000 ← too_few_trades(0tr)

Previous strategy had 0 trades. Making ADX trend filter stricter and adding volume confirmation. ---  Previous strategy had 0 trades. Making ADX trend filter stricter and adding volume confirmation. ---